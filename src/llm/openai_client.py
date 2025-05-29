"""OpenAI LLM client implementation."""

from typing import List, Dict, Any, Optional
import os
import logging
from openai import OpenAI
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMClient, Message, CompletionResponse, Role

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """OpenAI API client for LLM operations."""
    
    # Model context lengths
    MODEL_CONTEXT_LENGTHS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        organization: Optional[str] = None
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
            organization: Optional organization ID
        """
        self.model = model
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(
            api_key=api_key,
            organization=organization
        )
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"Initialized OpenAI client with model: {model}")
        logger.debug(f"Using API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    
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
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        # Extract additional parameters
        top_p = kwargs.get('top_p', 1.0)
        frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        presence_penalty = kwargs.get('presence_penalty', 0.0)
        stop = kwargs.get('stop', None)
        
        # Make API call
        logger.debug(f"Making OpenAI API call with model: {self.model}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )
        
        # Extract response
        choice = response.choices[0]
        usage = response.usage
        
        return CompletionResponse(
            content=choice.message.content,
            tokens_used=usage.total_tokens,
            model=response.model,
            finish_reason=choice.finish_reason,
            metadata={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "response_id": response.id
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model
    
    @property
    def max_context_length(self) -> int:
        """Get maximum context length."""
        return self.MODEL_CONTEXT_LENGTHS.get(self.model, 4096)
    
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
            "gpt-4o": {"prompt": 2.5, "completion": 10.0},
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
            "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
            "gpt-4": {"prompt": 30.0, "completion": 60.0},
            "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
        }
        
        model_pricing = pricing.get(self.model, {"prompt": 0.5, "completion": 1.5})
        
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
                "You are a helpful assistant that answers questions based on the provided context. "
                "Only use information from the context to answer. If the answer is not in the context, say so."
            ))
        
        # Add user message with context and question
        user_content = f"""Context:
{context}

Question: {question}

Please provide a clear and accurate answer based on the context above."""
        
        messages.append(Message(Role.USER, user_content))
        
        return messages