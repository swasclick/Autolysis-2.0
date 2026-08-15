from src.modules.error_module import APICallerError
from src.llm.providers.openai import OpenAIRequestParser
from src.llm.providers.gemini import GeminiRequestParser
from src.llm.providers.anthropic import AnthropicRequestParser
from src.llm.providers.ollama import OllamaRequestParser

'''
SUPPORTED PROVIDERS
- OpenAI
- Gemini
- Anthropic
- Ollama (local)
'''


class APICaller:
    
    def __init__(self, provider: str, creds: dict = {}):
        self.creds = creds
        self.provider = provider.lower().strip()
        self.providers = {
            "openai": OpenAIRequestParser,
            "anthropic": AnthropicRequestParser,
            "gemini": GeminiRequestParser,
            "ollama": OllamaRequestParser
        }
        
    def build_metadata_dict(self):
        
        metadata_dict = {
                "system_prompt": self.system_prompt,
                "user_prompt": self.user_prompt,
                "model": self.model,
                "temperature": self.temperature if self.temperature is not None else 0.7,
                "max_tokens": self.max_tokens if self.max_tokens is not None else 1000,
                "top_p": self.top_p if self.top_p is not None else 1.0,
                "stream": False,
            }
        return metadata_dict
    
    def build_request(self, metadata_dict):
        
        if self.provider not in self.providers.keys():
            raise APICallerError("Provider is not supported.")

        parser_class = self.providers[self.provider]
        builder = parser_class(self.creds, metadata_dict)
        return builder
    
    def send_request(self, builder):
        
        response = builder.call()
        return {
            'provider': self.provider,
            'response': response
        }
        
    def make_api_call(self, model: str = "", user_prompt: str= 'This is a default prompt respond by saying please provode a user prompt',
                 system_prompt: str = 'This is a default prompt respond by saying please provode a system prompt',
                 temperature: float = 0.7, max_tokens: int = 1000, top_p: float = 1.0):
        
        self.model = model
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        metadata_dict = self.build_metadata_dict()
        builder = self.build_request(metadata_dict= metadata_dict)
        response_dict = self.send_request(builder= builder)
        
        return response_dict
        
