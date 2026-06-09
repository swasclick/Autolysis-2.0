import requests

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
    