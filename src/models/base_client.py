from abc import ABC, abstractmethod


def build_response_instruction(
    expected_response_type: str | None,
    length_level: str | None = None
) -> str:
    instructions = []

    if expected_response_type and str(expected_response_type).strip():
        response_type = str(expected_response_type).strip().lower()

        if response_type == "code":
            instructions.append("Ответ должен содержать в основном только код без лишних пояснений.")
        elif response_type == "code_and_text":
            instructions.append("Ответ должен содержать код и краткое текстовое пояснение.")
        elif response_type == "text":
            instructions.append("Ответ должен быть текстовым, без кода, если код не требуется явно.")
        elif response_type == "structured_text":
            instructions.append("Ответ должен быть текстовым и хорошо структурированным.")
        elif response_type == "tests":
            instructions.append("Ответ должен содержать тесты или unit-тесты.")
        elif response_type == "test_cases":
            instructions.append("Ответ должен содержать набор тест-кейсов.")
        else:
            instructions.append("Формат ответа должен соответствовать ожидаемому типу результата.")

    if length_level and str(length_level).strip():
        level = str(length_level).strip().lower()

        if level == "short":
            instructions.append("Ответ должен быть кратким.")
        elif level == "medium":
            instructions.append("Ответ должен быть средней длины.")
        elif level == "long":
            instructions.append("Ответ должен быть подробным.")

    return " ".join(instructions).strip()


class BaseModelClient(ABC):
    model_name: str
    model_version: str

    @abstractmethod
    def generate_answer(
        self,
        prompt_text: str,
        role_text: str | None = None,
        expected_response_type: str | None = None,
        length_level: str | None = None,
    ) -> str:
        pass