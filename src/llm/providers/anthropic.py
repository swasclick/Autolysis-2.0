from anthropic import Anthropic
from src.modules.error_module import APICallerError
from modules.logging_module import Logger

logger = Logger().get_logger()

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
            logger.info('Got API response')
            return response.content[0].text
        except Exception as e:
            raise APICallerError(f'Error while making request to provider: {e}')

