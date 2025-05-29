"""Anthropic Claude LLM client implementation."""

from typing import List, Dict, Any, Optional
import os
import logging
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMClient, Message, CompletionResponse, Role

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Anthropic Claude API client for LLM operations."""
    
    # Model context lengths
    MODEL_CONTEXT_LENGTHS = {
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-2.1": 200000,
        "claude-2.0": 100000,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307"
    ):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            model: Model name to use
        """
        self.model = model
        
        # Initialize Anthropic client
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = Anthropic(api_key=api_key)
        
        logger.info(f"Initialized Anthropic client with model: {model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> CompletionResponse:
        """Generate a completion for a prompt."""
        # Convert to chat format
        messages = [Message(Role.USER, prompt)]
        return self.chat(messages, temperature, max_tokens, **kwargs)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> CompletionResponse:
        """Generate a chat completion."""
        # Separate system message from other messages
        system_message = None
        chat_messages = []
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_message = msg.content
            else:
                chat_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        
        # Extract additional parameters
        top_p = kwargs.get('top_p', 1.0)
        top_k = kwargs.get('top_k', None)
        stop_sequences = kwargs.get('stop', None)
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        if system_message:
            request_params["system"] = system_message
        if top_k is not None:
            request_params["top_k"] = top_k
        if stop_sequences:
            request_params["stop_sequences"] = stop_sequences
        
        # Make API call
        response = self.client.messages.create(**request_params)
        
        # Extract response content
        content = response.content[0].text if response.content else ""
        
        return CompletionResponse(
            content=content,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            model=response.model,
            finish_reason=response.stop_reason or "stop",
            metadata={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "response_id": response.id
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Note: This is an approximation. Anthropic doesn't provide
        a public tokenizer, so we estimate based on character count.
        """
        # Rough approximation: ~4 characters per token
        return len(text) // 4
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model
    
    @property
    def max_context_length(self) -> int:
        """Get maximum context length."""
        return self.MODEL_CONTEXT_LENGTHS.get(self.model, 100000)
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Estimate cost for tokens.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Estimated cost in USD
        """
        # Pricing as of 2024 (per 1M tokens)
        pricing = {
            "claude-3-5-sonnet-20241022": {"prompt": 3.0, "completion": 15.0},
            "claude-3-5-haiku-20241022": {"prompt": 1.0, "completion": 5.0},
            "claude-3-opus-20240229": {"prompt": 15.0, "completion": 75.0},
            "claude-3-sonnet-20240229": {"prompt": 3.0, "completion": 15.0},
            "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
        }
        
        model_pricing = pricing.get(self.model, {"prompt": 3.0, "completion": 15.0})
        
        prompt_cost = (prompt_tokens / 1_000_000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * model_pricing["completion"]
        
        return prompt_cost + completion_cost
    
    def create_rag_prompt(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> List[Message]:
        """
        Create messages for RAG prompt.
        
        Args:
            question: User question
            context: Retrieved context
            system_prompt: Optional system prompt
            
        Returns:
            List of messages
        """
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append(Message(Role.SYSTEM, system_prompt))
        else:
            messages.append(Message(
                Role.SYSTEM,
                "You are a helpful assistant that answers questions based on technical documentation. "
                "Only use information from the provided context to answer. "
                "If the answer is not in the context, clearly state that the information is not available."
            ))
        
        # Add user message with context and question
        user_content = f"""I'll provide you with some technical documentation context, followed by a question.

Context:
{context}

Question: {question}

Please provide a clear, accurate, and helpful answer based solely on the context above. If the context doesn't contain enough information to fully answer the question, please state what information is missing."""
        
        messages.append(Message(Role.USER, user_content))
        
        return messages