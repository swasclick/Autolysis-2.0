from openai import OpenAI
from error_module import APICallerError
from anthropic import Anthropic
from google import genai
import requests

'''
SUPPORTED PROVIDERS
- OpenAI
- Gemini
- Anthropic
- Ollama (local)
'''


class APICaller:
    
    def __init__(self, creds: dict, provider: str,
                 model: str = "", user_prompt: str= 'This is a default prompt respond by saying please provode a user prompt',
                 system_prompt: str = 'This is a default prompt respond by saying please provode a system prompt',
                 temperature: float = 0.7, max_tokens: int = 1000, top_p: float = 1.0
                 ):
        self.creds = creds
        self.provider = provider.lower().strip()
        self.model = model
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
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
        
    def make_api_call(self):
        
        metadata_dict = self.build_metadata_dict()
        builder = self.build_request(metadata_dict= metadata_dict)
        response_dict = self.send_request(builder= builder)
        
        return response_dict
        

class OpenAIRequestParser():
    def __init__(self, creds: dict, metadata_dict: dict):
        self.creds= creds
        self.metadata_dict = metadata_dict
        
    def call(self):
        '''Function to format the inputs into an openai compatible request'''
        client = OpenAI(api_key= self.creds["API_KEY"])
        payload = {
                "model": self.metadata_dict['model'],
                "input": [
                    {"role": "system", "content": self.metadata_dict['system_prompt']},
                    {"role": "user", "content": self.metadata_dict['user_prompt']}
                ],
                "temperature": self.metadata_dict.get("temperature", 0.7),
                "max_output_tokens": self.metadata_dict.get("max_tokens", 1000),
                "top_p": self.metadata_dict.get("top_p", 1.0)
            }
        try:
            response = client.responses.create(
                **payload
            )
            return response
        except Exception as e:
            raise APICallerError(f'Error while making request to provider: {e}')

class AnthropicRequestParser():
    def __init__(self, creds: dict, metadata_dict: dict):
        self.creds = creds
        self.metadata_dict = metadata_dict

    def call(self):
        client = Anthropic(api_key=self.creds["API_KEY"])
        payload = {
            'model':self.metadata_dict["model"],
            'system':self.metadata_dict["system_prompt"],
            'messages':[
                {
                    "role": "user",
                    "content": self.metadata_dict["user_prompt"]
                }
            ],
            'max_tokens':self.metadata_dict.get("max_tokens", 1000),
            'temperature':self.metadata_dict.get("temperature", 0.7)
        }
        try:
            response = client.messages.create(
                **payload
            )
            return response.content[0].text
        except Exception as e:
            raise APICallerError(f'Error while making request to provider: {e}')

class GeminiRequestParser():
    def __init__(self, creds: dict, metadata_dict: dict):
        self.creds = creds
        self.metadata_dict = metadata_dict

    def call(self):
        client = genai.Client(api_key=self.creds["API_KEY"])
        payload = {
            'model':self.metadata_dict["model"],
            'contents':[
                {
                    "role": "user",
                    "parts": [self.metadata_dict["user_prompt"]]
                }
            ],
            'config':{
                "temperature": self.metadata_dict.get("temperature", 0.7),
                "max_output_tokens": self.metadata_dict.get("max_tokens", 1000),
                "top_p": self.metadata_dict.get("top_p", 1.0)
            }
        }
        try:
            response = client.models.generate_content(
                **payload
            )
            return response.text
        except Exception as e:
            raise APICallerError(f'Error while making request to provider: {e}')

class OllamaRequestParser():
    def __init__(self, creds: dict, metadata_dict: dict):
        self.creds = creds
        self.metadata_dict = metadata_dict

    def call(self):
        payload = {
            "model": self.metadata_dict["model"],
            "messages": [
                {"role": "system", "content": self.metadata_dict["system_prompt"]},
                {"role": "user", "content": self.metadata_dict["user_prompt"]}
            ],
            "options": {
                "temperature": self.metadata_dict.get("temperature", 0.7),
                "top_p": self.metadata_dict.get("top_p", 1.0),
                "num_predict": self.metadata_dict.get("max_tokens", 1000)
            },
            "stream": False
        }

        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload
        )

        return response.json()["message"]["content"]
    