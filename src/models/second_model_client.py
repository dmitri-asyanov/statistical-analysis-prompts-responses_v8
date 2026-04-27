import os
import time
import uuid

import requests
from dotenv import load_dotenv

from src.models.base_client import BaseModelClient, build_response_instruction


load_dotenv()


GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"


class GigaChatClient(BaseModelClient):
    def __init__(self):
        self.auth_key = os.getenv("GIGACHAT_AUTH_KEY")
        self.scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.gigachat_model = os.getenv("GIGACHAT_MODEL", "GigaChat")

        verify_ssl_raw = os.getenv("GIGACHAT_VERIFY_SSL", "true").lower()
        self.verify_ssl = verify_ssl_raw not in {"0", "false", "no"}

        self.model_name = "gigachat"
        self.model_version = self.gigachat_model

        self._access_token: str | None = None
        self._expires_at_ms: int | None = None

        if not self.auth_key:
            raise ValueError("Не найден GIGACHAT_AUTH_KEY в .env")

    def _token_is_valid(self) -> bool:
        if not self._access_token or not self._expires_at_ms:
            return False

        current_time_ms = int(time.time() * 1000)
        safety_margin_ms = 60_000

        return current_time_ms < self._expires_at_ms - safety_margin_ms

    def _get_access_token(self) -> str:
        if self._token_is_valid():
            return self._access_token

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self.auth_key}",
        }

        data = {
            "scope": self.scope,
        }

        response = requests.post(
            GIGACHAT_AUTH_URL,
            headers=headers,
            data=data,
            timeout=60,
            verify=self.verify_ssl,
        )
        response.raise_for_status()

        token_data = response.json()

        self._access_token = token_data["access_token"]
        self._expires_at_ms = int(token_data["expires_at"])

        return self._access_token

    def generate_answer(
        self,
        prompt_text: str,
        role_text: str | None = None,
        expected_response_type: str | None = None,
        length_level: str | None = None,
    ) -> str:
        access_token = self._get_access_token()

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
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
                "content": " ".join(system_parts),
            })

        messages.append({
            "role": "user",
            "content": str(prompt_text).strip(),
        })

        payload = {
            "model": self.gigachat_model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 2000,
            "stream": False,
        }

        response = requests.post(
            GIGACHAT_CHAT_URL,
            headers=headers,
            json=payload,
            timeout=90,
            verify=self.verify_ssl,
        )
        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]