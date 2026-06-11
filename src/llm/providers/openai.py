from openai import OpenAI
from src.modules.error_module import APICallerError
from modules.logging_module import Logger

logger = Logger().get_logger()

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
            logger.info('Got API response')
            return response
        except Exception as e:
            raise APICallerError(f'Error while making request to provider: {e}')