import os
import re
import pandas as pd


TECH_TERMS = {
    "python", "function", "class", "algorithm", "test", "tests", "bug",
    "exception", "api", "json", "sql", "list", "dict", "loop",
    "docker", "unit", "pytest", "code", "debug", "repository"
}


def extract_code_blocks(text: str) -> list[str]:
    """
    Извлекает содержимое fenced code blocks: ```...```.
    """
    text = str(text)
    return re.findall(r"```(?:\w+)?\n?(.*?)```", text, flags=re.DOTALL)


def remove_code_blocks(text: str) -> str:
    """
    Удаляет fenced code blocks из текста.
    Используется для текстовых метрик, чтобы код не искажал читаемость.
    """
    text = str(text)
    return re.sub(r"```(?:\w+)?\n?.*?```", "", text, flags=re.DOTALL).strip()


def get_text_for_text_metrics(text: str) -> str:
    """
    Возвращает только обычный текст без code blocks.
    """
    return remove_code_blocks(text)


def count_words(text: str) -> int:
    text = get_text_for_text_metrics(text)
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def count_sentences(text: str) -> int:
    text = get_text_for_text_metrics(text)

    if not text:
        return 0

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def count_paragraphs(text: str) -> int:
    text = get_text_for_text_metrics(text)

    if not text:
        return 0

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return len(paragraphs)


def avg_sentence_length(text: str) -> float:
    sentences = count_sentences(text)
    words = count_words(text)

    return round(words / sentences, 4) if sentences > 0 else 0.0


def count_lists(text: str) -> int:
    text = get_text_for_text_metrics(text)
    lines = text.split("\n")

    return sum(
        1 for line in lines
        if re.match(r"^\s*([-*]|\d+\.)\s+", line)
    )


def has_code_block(text: str) -> int:
    text = str(text)
    return 1 if "```" in text else 0


def count_code_blocks(text: str) -> int:
    text = str(text)
    return text.count("```") // 2


def code_line_count(text: str) -> int:
    blocks = extract_code_blocks(text)

    return sum(
        len([line for line in block.splitlines() if line.strip()])
        for block in blocks
    )


def code_char_count(text: str) -> int:
    blocks = extract_code_blocks(text)
    return sum(len(block) for block in blocks)


def code_ratio(text: str) -> float:
    text = str(text)
    total_chars = len(text)

    if total_chars == 0:
        return 0.0

    return round(code_char_count(text) / total_chars, 4)


def text_char_count(text: str) -> int:
    return len(get_text_for_text_metrics(text))


def text_word_count(text: str) -> int:
    return count_words(text)


def text_ratio(text: str) -> float:
    text = str(text)
    total_chars = len(text)

    if total_chars == 0:
        return 0.0

    return round(text_char_count(text) / total_chars, 4)


def has_function_def(text: str) -> int:
    blocks = extract_code_blocks(text)
    code = "\n".join(blocks)

    return 1 if re.search(
        r"^\s*def\s+\w+\s*\(",
        code,
        flags=re.MULTILINE
    ) else 0


def has_import(text: str) -> int:
    blocks = extract_code_blocks(text)
    code = "\n".join(blocks)

    return 1 if re.search(
        r"^\s*(import\s+\w+|from\s+\w+.*import\s+)",
        code,
        flags=re.MULTILINE
    ) else 0


def has_assert(text: str) -> int:
    blocks = extract_code_blocks(text)
    code = "\n".join(blocks)

    return 1 if re.search(r"\bassert\b", code) else 0


def has_pytest(text: str) -> int:
    text = str(text).lower()
    return 1 if re.search(r"\bpytest\b", text) else 0


def has_unittest(text: str) -> int:
    text = str(text).lower()
    return 1 if re.search(r"\bunittest\b", text) else 0


def has_example(text: str) -> int:
    text = str(text).lower()

    patterns = [
        r"\bexample\b",
        r"\bexamples\b",
        r"\bfor example\b",
        r"\bнапример\b",
        r"\bпример\b",
        r"\bпримеры\b",
        r"\bпример использования\b",
    ]

    return 1 if any(re.search(pattern, text) for pattern in patterns) else 0


def has_complexity(text: str) -> int:
    text = str(text).lower()

    patterns = [
        r"\bo\([^)]+\)",
        r"\bbig[-\s]?o\b",
        r"\bcomplexity\b",
        r"\btime complexity\b",
        r"\bspace complexity\b",
        r"\bсложность\b",
        r"\bвременная сложность\b",
        r"\bпространственная сложность\b",
    ]

    return 1 if any(re.search(pattern, text) for pattern in patterns) else 0


def has_step_by_step(text: str) -> int:
    text = str(text).lower()

    patterns = [
        r"\bstep[-\s]?by[-\s]?step\b",
        r"\bstep\s+\d+\b",
        r"\bшаг\s+\d+\b",
        r"\bпошагово\b",
        r"\bпо шагам\b",
        r"\bсначала\b.*\bзатем\b",
        r"\bво-первых\b",
        r"\bво-вторых\b",
    ]

    return 1 if any(
        re.search(pattern, text, flags=re.DOTALL)
        for pattern in patterns
    ) else 0


def has_error_explanation(text: str) -> int:
    text = str(text).lower()

    patterns = [
        r"\berror\b",
        r"\bexception\b",
        r"\btraceback\b",
        r"\bbug\b",
        r"\bошибка\b",
        r"\bисключение\b",
        r"\bпочему возникает\b",
        r"\bпричина ошибки\b",
        r"\bпроблема в том\b",
    ]

    return 1 if any(re.search(pattern, text) for pattern in patterns) else 0


def has_fixed_code(text: str) -> int:
    text = str(text).lower()

    patterns = [
        r"\bfixed code\b",
        r"\bcorrected code\b",
        r"\bupdated code\b",
        r"\bисправленный код\b",
        r"\bисправленная версия\b",
        r"\bвариант исправления\b",
        r"\bвот исправленный\b",
        r"\bисправим\b",
    ]

    return 1 if has_code_block(text) and any(
        re.search(pattern, text)
        for pattern in patterns
    ) else 0


def lexical_diversity(text: str) -> float:
    text = get_text_for_text_metrics(text)
    words = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)

    if not words:
        return 0.0

    return round(len(set(words)) / len(words), 4)


def count_technical_terms(text: str) -> int:
    text = get_text_for_text_metrics(text)
    words = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)

    return sum(1 for word in words if word in TECH_TERMS)


def prompt_text_overlap(prompt: str, answer: str) -> float:
    """
    Считает пересечение слов prompt только с обычным текстом ответа,
    без code blocks.
    """
    prompt_words = set(
        re.findall(r"\b\w+\b", str(prompt).lower(), flags=re.UNICODE)
    )

    answer_text = get_text_for_text_metrics(answer)
    answer_words = set(
        re.findall(r"\b\w+\b", answer_text.lower(), flags=re.UNICODE)
    )

    if not prompt_words:
        return 0.0

    overlap = len(prompt_words & answer_words) / len(prompt_words)
    return round(overlap, 4)


def prompt_full_overlap(prompt: str, answer: str) -> float:
    """
    Считает пересечение слов prompt со всем ответом,
    включая обычный текст и code blocks.
    """
    prompt_words = set(
        re.findall(r"\b\w+\b", str(prompt).lower(), flags=re.UNICODE)
    )

    answer_words = set(
        re.findall(r"\b\w+\b", str(answer).lower(), flags=re.UNICODE)
    )

    if not prompt_words:
        return 0.0

    overlap = len(prompt_words & answer_words) / len(prompt_words)
    return round(overlap, 4)


def prompt_term_overlap(prompt: str, answer: str) -> float:
    """
    Старое имя признака оставлено для совместимости.
    Теперь оно эквивалентно prompt_text_overlap.
    """
    return prompt_text_overlap(prompt, answer)


def simple_readability_score(text: str) -> float:
    """
    Упрощённая прокси-метрика читаемости.
    Считается только по обычному тексту без code blocks.
    Чем длиннее предложения, тем ниже score.
    """
    avg_len = avg_sentence_length(text)

    if avg_len == 0:
        return 0.0

    score = 1 - (avg_len / 40)
    score = max(0.0, min(1.0, score))

    return round(score, 4)


def simple_structure_score(text: str) -> float:
    """
    Упрощённая формальная оценка структуры ответа.

    Абзацы, списки и предложения считаются только по обычному тексту.
    Code block учитывается отдельно как структурный элемент.
    """
    score = 0.0

    if count_paragraphs(text) > 1:
        score += 0.25

    if count_lists(text) > 0:
        score += 0.25

    if has_code_block(text):
        score += 0.25

    if count_sentences(text) >= 3:
        score += 0.25

    return round(min(score, 1.0), 4)


def extract_features(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("answers_raw.csv пустой")

    required_columns = {"answer_id", "prompt_id", "answer_text"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"В answers_raw.csv отсутствуют столбцы: {missing}"
        )

    if "prompt_text" not in df.columns:
        df["prompt_text"] = ""

    df["char_count"] = df["answer_text"].astype(str).apply(len)

    df["text_char_count"] = df["answer_text"].apply(text_char_count)
    df["text_word_count"] = df["answer_text"].apply(text_word_count)
    df["text_ratio"] = df["answer_text"].apply(text_ratio)

    df["word_count"] = df["answer_text"].apply(count_words)
    df["sentence_count"] = df["answer_text"].apply(count_sentences)
    df["paragraph_count"] = df["answer_text"].apply(count_paragraphs)
    df["avg_sentence_length"] = df["answer_text"].apply(avg_sentence_length)
    df["list_count"] = df["answer_text"].apply(count_lists)

    df["has_code"] = df["answer_text"].apply(has_code_block)
    df["code_block_count"] = df["answer_text"].apply(count_code_blocks)
    df["code_line_count"] = df["answer_text"].apply(code_line_count)
    df["code_char_count"] = df["answer_text"].apply(code_char_count)
    df["code_ratio"] = df["answer_text"].apply(code_ratio)

    df["has_function_def"] = df["answer_text"].apply(has_function_def)
    df["has_import"] = df["answer_text"].apply(has_import)
    df["has_assert"] = df["answer_text"].apply(has_assert)
    df["has_pytest"] = df["answer_text"].apply(has_pytest)
    df["has_unittest"] = df["answer_text"].apply(has_unittest)
    df["has_example"] = df["answer_text"].apply(has_example)
    df["has_complexity"] = df["answer_text"].apply(has_complexity)
    df["has_step_by_step"] = df["answer_text"].apply(has_step_by_step)
    df["has_error_explanation"] = df["answer_text"].apply(
        has_error_explanation
    )
    df["has_fixed_code"] = df["answer_text"].apply(has_fixed_code)

    df["technical_terms_count"] = df["answer_text"].apply(
        count_technical_terms
    )
    df["readability_score"] = df["answer_text"].apply(
        simple_readability_score
    )
    df["lexical_diversity"] = df["answer_text"].apply(lexical_diversity)
    df["structure_score"] = df["answer_text"].apply(simple_structure_score)

    df["prompt_text_overlap"] = df.apply(
        lambda row: prompt_text_overlap(
            row["prompt_text"],
            row["answer_text"]
        ),
        axis=1
    )

    df["prompt_full_overlap"] = df.apply(
        lambda row: prompt_full_overlap(
            row["prompt_text"],
            row["answer_text"]
        ),
        axis=1
    )


    preferred_order = [
        "answer_id",
        "prompt_id",
        "category",
        "category_label",
        "length_level",
        "expected_response_type",
        "has_context",
        "has_constraints",
        "role",
        "role_text",
        "context_text",
        "constraints_text",
        "template_text",
        "variable_values",
        "prompt_text",
        "model_name",
        "model_version",
        "answer_text",
        "generated_at",
        "response_time_sec",

        "char_count",
        "text_char_count",
        "text_word_count",
        "text_ratio",
        "word_count",
        "sentence_count",
        "paragraph_count",
        "avg_sentence_length",
        "list_count",

        "has_code",
        "code_block_count",
        "code_line_count",
        "code_char_count",
        "code_ratio",

        "has_function_def",
        "has_import",
        "has_assert",
        "has_pytest",
        "has_unittest",
        "has_example",
        "has_complexity",
        "has_step_by_step",
        "has_error_explanation",
        "has_fixed_code",

        "technical_terms_count",
        "readability_score",
        "lexical_diversity",
        "structure_score",

        "prompt_text_overlap",
        "prompt_full_overlap",

        "formal_clarity_score",
        "formal_structure_score",
        "formal_utility_score",
        "formal_relevance_score",
        "formal_total_score",
        "task_score",
        "judge_relevance",
        "judge_completeness",
        "judge_clarity",
        "judge_logic",
        "judge_utility",
        "judge_comment",
        "judge_total_score",
        "final_score"
    ]

    for col in preferred_order:
        if col not in df.columns:
            df[col] = None

    existing_columns = [col for col in preferred_order if col in df.columns]
    remaining_columns = [
        col for col in df.columns
        if col not in existing_columns
    ]

    df = df[existing_columns + remaining_columns]

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Обработанные ответы сохранены в {output_path}")


if __name__ == "__main__":
    extract_features(
        input_path="data/raw/answers_raw.csv",
        output_path="data/processed/answers.csv"
    )