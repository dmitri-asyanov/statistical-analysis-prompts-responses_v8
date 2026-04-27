import re


TECH_TERMS = {
    "python", "код", "функция", "алгоритм", "тест", "тесты",
    "unit", "json", "api", "ошибка", "исключение", "список",
    "словарь", "строка", "число", "сортировка", "поиск",
    "сложность", "библиотека", "зависимость", "backend",
    "сервис", "модуль", "pytest", "debug", "отладка"
}


DIRECTIVE_WORDS = {
    "напиши", "реализуй", "исправь", "найди", "объясни",
    "опиши", "укажи", "добавь", "составь", "раздели",
    "покажи", "проанализируй", "приведи", "используй",
    "избегай", "сохрани"
}


FORMAT_WORDS = {
    "кратко", "подробно", "структурированным", "структурированный",
    "поясни", "объясни", "комментарии", "пример", "код",
    "unit-тесты", "тест-кейсы", "позитивные", "негативные",
    "граничные", "сложность"
}


CONSTRAINT_MARKERS = {
    "ограничения", "используй только", "не используй",
    "избегай", "сохрани", "добавь", "раздели",
    "укажи", "при необходимости"
}


INSTRUCTION_MARKERS = {
    "напиши", "реализуй", "исправь", "найди", "объясни",
    "опиши", "составь", "проанализируй", "покажи",
    "приведи", "добавь", "укажи"
}


CODE_PATTERNS = [
    r"```",
    r"\bdef\s+\w+\s*\(",
    r"\bclass\s+\w+",
    r"\breturn\b",
    r"\bprint\s*\(",
    r"\bfor\s+\w+\s+in\s+",
    r"\bif\s+.+:",
    r"\bimport\s+\w+",
    r"\bfrom\s+\w+\s+import\b",
    r"\w+\s*=\s*.+",
]


def _to_text(text: str) -> str:
    return "" if text is None else str(text)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", _to_text(text), flags=re.UNICODE))


def count_sentences(text: str) -> int:
    parts = re.split(r"[.!?]+", _to_text(text))
    return len([part.strip() for part in parts if part.strip()])


def count_lines(text: str) -> int:
    lines = _to_text(text).splitlines()
    return len([line for line in lines if line.strip()])


def count_paragraphs(text: str) -> int:
    paragraphs = re.split(r"\n\s*\n", _to_text(text).strip())
    return len([p for p in paragraphs if p.strip()])


def count_list_items(text: str) -> int:
    lines = _to_text(text).splitlines()
    return sum(
        1
        for line in lines
        if re.match(r"^\s*(-|\*|\d+\.)\s+", line)
    )


def has_question_form(text: str) -> int:
    return 1 if "?" in _to_text(text) else 0


def count_technical_terms(text: str) -> int:
    words = re.findall(r"\b\w+\b", _to_text(text).lower(), flags=re.UNICODE)
    return sum(1 for word in words if word in TECH_TERMS)


def has_code_in_prompt(text: str) -> int:
    text = _to_text(text)

    for pattern in CODE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE):
            return 1

    return 0


def has_format_instruction(text: str) -> int:
    lowered = _to_text(text).lower()
    return 1 if any(word in lowered for word in FORMAT_WORDS) else 0


def count_directive_words(text: str) -> int:
    words = re.findall(r"\b\w+\b", _to_text(text).lower(), flags=re.UNICODE)
    return sum(1 for word in words if word in DIRECTIVE_WORDS)


def count_constraints(text: str) -> int:
    text = _to_text(text)
    lowered = text.lower()

    count = 0

    # Явный блок ограничений
    if "ограничения:" in lowered:
        lines = text.splitlines()
        inside_constraints = False

        for line in lines:
            stripped = line.strip()
            lower_line = stripped.lower()

            if lower_line.startswith("ограничения:"):
                inside_constraints = True
                continue

            if inside_constraints:
                if re.match(r"^-\s+", stripped):
                    count += 1
                elif stripped == "":
                    continue
                else:
                    inside_constraints = False

    # Дополнительная страховка по ключевым фразам
    for marker in CONSTRAINT_MARKERS:
        if marker in lowered:
            count += 1

    return count


def count_instructions(text: str) -> int:
    lowered = _to_text(text).lower()

    count = 0

    for marker in INSTRUCTION_MARKERS:
        count += len(re.findall(rf"\b{re.escape(marker)}\b", lowered, flags=re.UNICODE))

    return count


def calc_structure_score(text: str) -> float:
    """
    Оценка структурированности промпта от 0 до 1.

    Учитываются:
    - наличие нескольких абзацев;
    - наличие нескольких строк;
    - наличие списка ограничений;
    - наличие роли/контекста/формата;
    - наличие явной инструкции.
    """
    text = _to_text(text)
    lowered = text.lower()

    score = 0.0

    if count_paragraphs(text) >= 2:
        score += 0.20

    if count_lines(text) >= 3:
        score += 0.20

    if count_list_items(text) > 0:
        score += 0.20

    if "контекст:" in lowered:
        score += 0.15

    if "ограничения:" in lowered:
        score += 0.15

    if count_instructions(text) > 0:
        score += 0.10

    return round(min(score, 1.0), 4)


def calc_specificity_score(text: str, meta: dict | None = None) -> float:
    """
    Оценка конкретности промпта от 0 до 1.

    Учитываются:
    - наличие контекста;
    - наличие ограничений;
    - наличие формата ответа;
    - наличие технических терминов;
    - наличие кода;
    - количество директивных слов;
    - достаточная длина промпта.
    """
    meta = meta or {}
    text = _to_text(text)

    word_count = count_words(text)
    tech_count = count_technical_terms(text)
    directive_count = count_directive_words(text)

    score = 0.0

    if int(meta.get("has_context", 0)) == 1:
        score += 0.20

    if int(meta.get("has_constraints", 0)) == 1:
        score += 0.20

    if has_format_instruction(text):
        score += 0.15

    if tech_count >= 1:
        score += 0.15

    if has_code_in_prompt(text):
        score += 0.10

    if directive_count >= 1:
        score += 0.10

    if word_count >= 20:
        score += 0.10

    return round(min(score, 1.0), 4)


def extract_prompt_features(text: str, meta: dict | None = None) -> dict:
    meta = meta or {}
    text = _to_text(text)

    role_value = str(meta.get("role", "")).strip()
    role_text = str(meta.get("role_text", "")).strip()

    has_role = 1 if role_value not in {"", "none"} or role_text else 0

    return {
        "char_count": len(text),
        "word_count": count_words(text),
        "sentence_count": count_sentences(text),
        "prompt_line_count": count_lines(text),
        "prompt_paragraph_count": count_paragraphs(text),

        "has_question_form": has_question_form(text),
        "has_role": has_role,
        "has_format_instruction": has_format_instruction(text),
        "has_code_in_prompt": has_code_in_prompt(text),

        "constraint_count": count_constraints(text),
        "instruction_count": count_instructions(text),
        "directive_words_count": count_directive_words(text),

        "technical_terms_count": count_technical_terms(text),
        "structure_score": calc_structure_score(text),
        "specificity_score": calc_specificity_score(text, meta),
    }