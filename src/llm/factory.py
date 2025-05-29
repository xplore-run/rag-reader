"""Factory for creating LLM client instances."""

from typing import Optional, Dict, Any
import os
from .base import LLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient


class LLMFactory:
    """Factory for creating LLM client instances."""
    
    @staticmethod
    def create(
        provider: str = "openai",
        **kwargs
    ) -> LLMClient:
        """
        Create an LLM client instance.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'ollama')
            **kwargs: Additional arguments for the specific client
            
        Returns:
            LLMClient instance
        """
        provider = provider.lower()
        
        if provider == "openai":
            return LLMFactory._create_openai(**kwargs)
        elif provider == "anthropic":
            return LLMFactory._create_anthropic(**kwargs)
        elif provider == "ollama":
            return LLMFactory._create_ollama(**kwargs)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    @staticmethod
    def _create_openai(**kwargs) -> OpenAIClient:
        """Create OpenAI client with default settings."""
        api_key = kwargs.get('api_key', os.getenv('OPENAI_API_KEY'))
        model = kwargs.get('model', os.getenv('OPENAI_MODEL', 'gpt-4o-mini'))
        organization = kwargs.get('organization', os.getenv('OPENAI_ORGANIZATION'))
        
        return OpenAIClient(
            api_key=api_key,
            model=model,
            organization=organization
        )
    
    @staticmethod
    def _create_anthropic(**kwargs) -> AnthropicClient:
        """Create Anthropic client with default settings."""
        api_key = kwargs.get('api_key', os.getenv('ANTHROPIC_API_KEY'))
        model = kwargs.get('model', os.getenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307'))
        
        return AnthropicClient(
            api_key=api_key,
            model=model
        )
    
    @staticmethod
    def _create_ollama(**kwargs):
        """Create Ollama client (to be implemented)."""
        # Import here to avoid dependency if not using Ollama
        from .ollama_client import OllamaClient
        
        base_url = kwargs.get('base_url', 'http://localhost:11434')
        model = kwargs.get('model', 'llama2')
        
        return OllamaClient(
            base_url=base_url,
            model=model
        )
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> LLMClient:
        """
        Create LLM client from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'provider' and config keys
            
        Returns:
            LLMClient instance
        """
        provider = config.get('provider', 'openai')
        client_config = config.get('config', {})
        
        return LLMFactory.create(provider, **client_config)
    
    @staticmethod
    def get_recommended_config(use_case: str = "general") -> Dict[str, Any]:
        """
        Get recommended LLM configuration for a use case.
        
        Args:
            use_case: Use case type
            
        Returns:
            Configuration dictionary
        """
        recommendations = {
            "general": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini"
                }
            },
            "high_quality": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o"
                }
            },
            "fast": {
                "provider": "anthropic",
                "config": {
                    "model": "claude-3-haiku-20240307"
                }
            },
            "long_context": {
                "provider": "anthropic",
                "config": {
                    "model": "claude-3-5-sonnet-20241022"
                }
            },
            "local": {
                "provider": "ollama",
                "config": {
                    "model": "llama2"
                }
            },
            "budget": {
                "provider": "openai",
                "config": {
                    "model": "gpt-3.5-turbo"
                }
            }
        }
        
        return recommendations.get(use_case, recommendations["general"])
    
    @staticmethod
    # todo - add more models and more use cases and ability to switch between models from cli.
    def compare_models() -> Dict[str, Dict[str, Any]]:
        """Get comparison of available models."""
        return {
            "gpt-4o": {
                "provider": "openai",
                "context_length": 128000,
                "strengths": ["High quality", "Multimodal", "Fast"],
                "cost": "Medium",
                "use_cases": ["Complex reasoning", "Code generation"]
            },
            "gpt-4o-mini": {
                "provider": "openai",
                "context_length": 128000,
                "strengths": ["Good quality", "Fast", "Cost-effective"],
                "cost": "Low",
                "use_cases": ["General Q&A", "Simple tasks"]
            },
            "claude-3-5-sonnet": {
                "provider": "anthropic",
                "context_length": 200000,
                "strengths": ["Very long context", "High quality", "Good reasoning"],
                "cost": "Medium",
                "use_cases": ["Document analysis", "Complex tasks"]
            },
            "claude-3-haiku": {
                "provider": "anthropic",
                "context_length": 200000,
                "strengths": ["Fast", "Long context", "Cheap"],
                "cost": "Very Low",
                "use_cases": ["Quick responses", "High volume"]
            }
        }