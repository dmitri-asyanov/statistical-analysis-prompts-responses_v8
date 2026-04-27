import re
from typing import Iterable

TECHNICAL_TERMS = {
    "python",
    "json",
    "backend",
    "unit",
    "тест",
    "тесты",
    "алгоритм",
    "функция",
    "код",
    "словарь",
    "список",
    "библиотека",
    "массив",
    "сортировка",
    "поиск",
    "ошибка",
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b[\w\-]+\b", text.lower(), flags=re.UNICODE)


def count_words(text: str) -> int:
    return len(tokenize_words(text))


def count_sentences(text: str) -> int:
    parts = re.split(r"[.!?]+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


def has_question_form(text: str) -> int:
    return int("?" in text)


def count_technical_terms(text: str, terms: Iterable[str] | None = None) -> int:
    words = set(tokenize_words(text))
    dictionary = set(terms) if terms is not None else TECHNICAL_TERMS
    return sum(1 for term in dictionary if term in words)