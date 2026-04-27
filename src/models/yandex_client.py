import os

import requests
from dotenv import load_dotenv

from src.models.base_client import BaseModelClient, build_response_instruction


load_dotenv()


YANDEX_GPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


class YandexGPTClient(BaseModelClient):
    def __init__(self):
        self.api_key = os.getenv("YANDEX_API_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.model_uri = os.getenv(
            "YANDEX_MODEL_URI",
            f"gpt://{self.folder_id}/yandexgpt-lite/latest"
        )

        self.model_name = "yandexgpt-lite"
        self.model_version = "latest"

        if not self.api_key:
            raise ValueError("Не найден YANDEX_API_KEY в .env")

        if not self.folder_id:
            raise ValueError("Не найден YANDEX_FOLDER_ID в .env")

    def generate_answer(
        self,
        prompt_text: str,
        role_text: str | None = None,
        expected_response_type: str | None = None,
        length_level: str | None = None,
    ) -> str:
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []

        system_parts = []

        if role_text and str(role_text).strip():
            system_parts.append(str(role_text).strip())

        response_instruction = build_response_instruction(
            expected_response_type=expected_response_type,
            length_level=length_level,
        )

        if response_instruction:
            system_parts.append(response_instruction)

        if system_parts:
            messages.append({
                "role": "system",
                "text": " ".join(system_parts),
            })

        messages.append({
            "role": "user",
            "text": str(prompt_text).strip(),
        })

        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": 2000,
            },
            "messages": messages,
        }

        response = requests.post(
            YANDEX_GPT_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        return data["result"]["alternatives"][0]["message"]["text"]