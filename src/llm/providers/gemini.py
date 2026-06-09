from google import genai
from src.modules.error_module import APICallerError

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