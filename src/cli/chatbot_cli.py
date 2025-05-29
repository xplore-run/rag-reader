#!/usr/bin/env python3
"""Command-line interface for the RAG chatbot."""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich import print as rprint
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.embeddings.factory import EmbeddingFactory
from src.vector_store.factory import VectorStoreFactory
from src.llm.factory import LLMFactory
from src.rag.retriever import RAGRetriever
from src.rag.generator import RAGGenerator
from src.utils.config import get_settings
from src.utils.logger import setup_logging, logger

# Initialize Rich console
console = Console()


class ChatbotCLI:
    """CLI interface for the RAG chatbot."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the chatbot CLI."""
        self.settings = get_settings(config_path)
        self.console = Console()
        
        # Set up logging
        setup_logging(
            log_level=self.settings.log_level,
            log_file=self.settings.log_file
        )
        
        # Initialize components
        self._initialize_components()
        
        # Chat history
        self.chat_history: List[Dict[str, str]] = []
    
    def _initialize_components(self):
        """Initialize RAG components."""
        with self.console.status("[bold green]Initializing chatbot...") as status:
            # Initialize embedding model
            status.update("Loading embedding model...")
            self.embedding_model = EmbeddingFactory.create(
                model_type=self.settings.embedding.type,
                model_name=self.settings.embedding.model_name,
                device=self.settings.embedding.device
            )
            
            # Initialize vector store
            status.update("Loading vector store...")
            self.vector_store = VectorStoreFactory.create(
                store_type=self.settings.vector_store.type,
                collection_name=self.settings.vector_store.collection_name,
                persist_directory=self.settings.vector_store.persist_directory
            )
            
            # Initialize LLM client
            status.update("Loading language model...")
            # Get the appropriate API key based on provider
            llm_kwargs = {
                'provider': self.settings.llm.provider,
                'model': self.settings.llm.model
            }
            
            if self.settings.llm.provider == 'openai' and self.settings.openai_api_key:
                llm_kwargs['api_key'] = self.settings.openai_api_key
            elif self.settings.llm.provider == 'anthropic' and self.settings.anthropic_api_key:
                llm_kwargs['api_key'] = self.settings.anthropic_api_key
                
            self.llm_client = LLMFactory.create(**llm_kwargs)
            
            # Initialize RAG components
            status.update("Setting up RAG pipeline...")
            self.retriever = RAGRetriever(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                rerank=self.settings.rag.rerank,
                include_neighbors=self.settings.rag.include_neighbors
            )
            
            self.generator = RAGGenerator(
                llm_client=self.llm_client,
                prompt_template=self.settings.rag.prompt_template,
                max_context_length=3000,
                include_sources=True
            )
        
        # Check if vector store has documents
        doc_count = self.vector_store.count()
        if doc_count == 0:
            self.console.print(
                "[bold red]Warning:[/bold red] No documents found in vector store. "
                "Please run the indexing script first."
            )
        else:
            self.console.print(
                f"[bold green]✓[/bold green] Loaded {doc_count} document chunks"
            )
    
    def ask_question(self, question: str, show_sources: bool = True) -> str:
        """
        Ask a question and get an answer.
        
        Args:
            question: User question
            show_sources: Whether to show source documents
            
        Returns:
            Generated answer
        """
        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            # Retrieve relevant documents
            task = progress.add_task("Searching documents...", total=None)
            
            retrieval_results = self.retriever.retrieve(
                query=question,
                k=self.settings.rag.top_k,
                score_threshold=self.settings.rag.score_threshold
            )
            
            progress.update(task, description="Generating answer...")
            
            # Generate answer
            answer = self.generator.generate(
                question=question,
                retrieval_results=retrieval_results
            )
        
        # Display answer
        self.console.print()
        self.console.print(Panel(
            Markdown(answer.answer),
            title="[bold cyan]Answer[/bold cyan]",
            border_style="cyan"
        ))
        
        # Display sources if requested
        if show_sources and answer.sources:
            self._display_sources(answer.sources)
        
        # Display confidence
        confidence_color = "green" if answer.confidence > 0.8 else "yellow" if answer.confidence > 0.6 else "red"
        self.console.print(
            f"\n[{confidence_color}]Confidence: {answer.confidence:.2%}[/{confidence_color}]"
        )
        
        return answer.answer
    
    def _display_sources(self, sources: List[Dict[str, Any]]):
        """Display source documents in a table."""
        table = Table(title="Source Documents", show_header=True)
        table.add_column("Document", style="cyan")
        table.add_column("Relevance", style="green")
        table.add_column("Chunks", style="yellow")
        
        for source in sources:
            table.add_row(
                source['document'],
                f"{source['relevance_score']:.2%}",
                str(len(source['chunks']))
            )
        
        self.console.print()
        self.console.print(table)
    
    def interactive_chat(self):
        """Run interactive chat session."""
        self.console.print(Panel(
            "[bold cyan]Technical Documentation RAG Chatbot[/bold cyan]\n"
            "Ask questions about your technical documents.\n"
            "Type 'help' for commands or 'exit' to quit.",
            title="Welcome",
            border_style="cyan"
        ))
        
        while True:
            try:
                # Get user input
                question = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                # Handle commands
                if question.lower() in ['exit', 'quit', 'q']:
                    if Confirm.ask("Are you sure you want to exit?"):
                        self.console.print("[bold green]Goodbye![/bold green]")
                        break
                    continue
                
                elif question.lower() == 'help':
                    self._show_help()
                    continue
                
                elif question.lower() == 'history':
                    self._show_history()
                    continue
                
                elif question.lower() == 'clear':
                    self.console.clear()
                    continue
                
                elif question.lower() == 'stats':
                    self._show_stats()
                    continue
                
                elif question.lower().startswith('config'):
                    self._show_config()
                    continue
                
                # Process question
                answer = self.ask_question(question)
                
                # Add to history
                self.chat_history.append({
                    'question': question,
                    'answer': answer,
                    'timestamp': time.time()
                })
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
## Available Commands

- **exit/quit/q**: Exit the chatbot
- **help**: Show this help message
- **history**: Show chat history
- **clear**: Clear the screen
- **stats**: Show system statistics
- **config**: Show current configuration

## Tips

- Ask specific questions about your technical documents
- The chatbot only answers based on indexed documents
- Use natural language for best results
"""
        self.console.print(Panel(
            Markdown(help_text),
            title="[bold cyan]Help[/bold cyan]",
            border_style="cyan"
        ))
    
    def _show_history(self):
        """Show chat history."""
        if not self.chat_history:
            self.console.print("[yellow]No chat history yet.[/yellow]")
            return
        
        for i, entry in enumerate(self.chat_history[-10:], 1):
            self.console.print(f"\n[bold cyan]Q{i}:[/bold cyan] {entry['question']}")
            self.console.print(f"[bold green]A{i}:[/bold green] {entry['answer'][:200]}...")
    
    def _show_stats(self):
        """Show system statistics."""
        stats = {
            "Documents in store": self.vector_store.count(),
            "Embedding model": self.settings.embedding.model_name or self.settings.embedding.type,
            "LLM model": self.settings.llm.model or self.settings.llm.provider,
            "Vector store": self.settings.vector_store.type,
            "Questions asked": len(self.chat_history)
        }
        
        table = Table(title="System Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(key, str(value))
        
        self.console.print()
        self.console.print(table)
    
    def _show_config(self):
        """Show current configuration."""
        config_dict = self.settings.to_dict()
        
        # Format as YAML-like output
        from rich.tree import Tree
        
        tree = Tree("[bold cyan]Configuration[/bold cyan]")
        
        def add_items(parent, items):
            for key, value in items.items():
                if isinstance(value, dict):
                    branch = parent.add(f"[cyan]{key}[/cyan]")
                    add_items(branch, value)
                else:
                    parent.add(f"[cyan]{key}:[/cyan] [green]{value}[/green]")
        
        add_items(tree, config_dict)
        self.console.print()
        self.console.print(tree)


@click.command()
@click.option(
    '--config', '-c',
    help='Path to configuration file',
    type=click.Path(exists=True)
)
@click.option(
    '--question', '-q',
    help='Ask a single question and exit'
)
@click.option(
    '--no-sources', 
    is_flag=True,
    help='Do not show source documents'
)
def main(config: Optional[str], question: Optional[str], no_sources: bool):
    """Technical Documentation RAG Chatbot CLI."""
    try:
        # Initialize chatbot
        chatbot = ChatbotCLI(config_path=config)
        
        if question:
            # Single question mode
            answer = chatbot.ask_question(question, show_sources=not no_sources)
        else:
            # Interactive mode
            chatbot.interactive_chat()
    
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()