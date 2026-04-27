import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "processed" / "answers.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "answers.csv"


ERROR_RESPONSE_PATTERNS = [
    r"^\s*ERROR\s*:",
    r"^\s*Traceback \(most recent call last\)",
    r"^\s*ConnectionError",
    r"^\s*Timeout",
    r"^\s*SSLError",
    r"^\s*Max retries exceeded",
    r"^\s*API key",
    r"^\s*Unauthorized",
]


PYTHON_CODE_PATTERNS = [
    r"\bdef\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
    r"\bclass\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[:\(]",
    r"\breturn\b",
    r"\bfor\s+\w+\s+in\s+",
    r"\bif\s+.+:",
    r"\btry\s*:",
    r"\bexcept\b",
    r"\bimport\s+[a-zA-Z_]",
    r"\bfrom\s+[a-zA-Z_].+\bimport\b",
    r"\bprint\s*\(",
    r"\w+\s*=\s*.+",
    r"\w+\.get\s*\(",
    r"\[[^\]]+\]",
    r"\{[^{}]*:[^{}]*\}",
]


ERROR_WORDS = [
    "ошибка", "исключение", "причина", "неверно", "некорректно",
    "typeerror", "indexerror", "syntaxerror", "valueerror",
    "zerodivisionerror", "keyerror", "attributeerror", "nameerror",
]


FIX_WORDS = [
    "исправ", "исправленный", "исправленная", "исправленная версия",
    "нужно заменить", "заменим", "теперь", "корректный вариант",
    "правильный вариант", "fixed", "fix",
]


ALGORITHM_WORDS = [
    "алгоритм", "работает", "принцип", "идея", "суть", "шаг",
    "сначала", "затем", "далее", "после этого", "на каждом шаге",
    "в итоге", "возвращает", "находит", "проверяет",
]


COMPLEXITY_PATTERNS = [
    r"\bO\s*\([^)]+\)",
    "временная сложность",
    "сложность",
    "линейная",
    "логарифмическая",
    "квадратичная",
    "константная",
]


EXAMPLE_WORDS = [
    "пример", "например", "допустим", "рассмотрим", "вход", "выход",
    "input", "output",
]


TEST_WORDS = [
    "assert", "pytest", "unittest", "test_", "тест", "тест-кейс",
    "проверяет", "ожидаемый результат", "expected", "actual",
]


POSITIVE_TEST_WORDS = [
    "позитив", "корректн", "валидн", "обычный случай", "normal case",
    "valid", "success",
]


NEGATIVE_TEST_WORDS = [
    "негатив", "ошибк", "исключение", "некорректн", "invalid",
    "exception", "raises", "pytest.raises", "деление на ноль",
]


BOUNDARY_TEST_WORDS = [
    "гранич", "краевой", "пустой", "ноль", "none", "null",
    "минимальн", "максимальн", "edge", "boundary", "empty",
]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_lower(value: Any) -> str:
    return normalize_text(value).lower()


def to_float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def bool_feature(row: pd.Series, column: str) -> bool:
    return to_int(row.get(column, 0)) == 1


def contains_any(text: str, words: list[str]) -> bool:
    text_lower = text.lower()
    return any(word.lower() in text_lower for word in words)


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE) for pattern in patterns)


def clamp_score(score: float) -> float:
    return round(max(0.0, min(1.0, score)), 4)


def get_answer_text(row: pd.Series) -> str:
    return normalize_text(row.get("answer_text", ""))


def get_prompt_text(row: pd.Series) -> str:
    return normalize_text(row.get("prompt_text", ""))


def get_expected_response_type(row: pd.Series) -> str:
    return normalize_lower(row.get("expected_response_type", ""))


def get_length_level(row: pd.Series) -> str:
    return normalize_lower(row.get("length_level", ""))


def is_error_response(text: str) -> bool:
    """
    Техническая ошибка сбора ответа.
    Важно: traceback внутри объяснения bug_fixing не считается ошибкой API.
    """
    text = normalize_text(text)
    if not text:
        return True

    if re.search(r"^\s*ERROR\s*:", text, flags=re.IGNORECASE):
        return True

    if len(text) <= 500 and matches_any(text, ERROR_RESPONSE_PATTERNS):
        return True

    return False


def has_code(row: pd.Series, text: str) -> bool:
    return (
        bool_feature(row, "has_code")
        or to_int(row.get("code_block_count", 0)) > 0
        or to_int(row.get("code_line_count", 0)) > 0
        or matches_any(text, PYTHON_CODE_PATTERNS)
    )


def has_function_def(row: pd.Series, text: str) -> bool:
    return bool_feature(row, "has_function_def") or bool(re.search(
        r"\bdef\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
        text,
    ))


def has_assert_or_framework(row: pd.Series, text: str) -> bool:
    return (
        bool_feature(row, "has_assert")
        or bool_feature(row, "has_pytest")
        or bool_feature(row, "has_unittest")
        or contains_any(text, ["assert", "pytest", "unittest", "pytest.raises", "test_"])
    )


def has_explanation_text(row: pd.Series, text: str) -> bool:
    text_word_count = to_int(row.get("text_word_count", 0))
    sentence_count = to_int(row.get("sentence_count", 0))
    paragraph_count = to_int(row.get("paragraph_count", 0))

    if text_word_count >= 12:
        return True

    if sentence_count >= 2 or paragraph_count >= 2:
        return True

    return contains_any(text, [
        "объяснение", "логика", "суть", "причина", "потому что",
        "работает так", "решение", "пояснение",
    ])


def explanation_required(row: pd.Series) -> bool:
    expected = get_expected_response_type(row)
    length = get_length_level(row)

    return expected in {"code_and_text", "structured_text"} or length in {"medium", "long"}


def has_relevance(row: pd.Series, min_overlap: float = 0.08) -> bool:
    prompt_text_overlap = to_float(row.get("prompt_text_overlap", 0.0))
    prompt_full_overlap = to_float(row.get("prompt_full_overlap", 0.0))
    old_prompt_overlap = to_float(row.get("prompt_term_overlap", 0.0))

    return max(prompt_text_overlap, prompt_full_overlap, old_prompt_overlap) >= min_overlap


def count_code_like_lines(text: str) -> int:
    lines = [line.strip() for line in normalize_text(text).split("\n") if line.strip()]

    code_like = 0
    for line in lines:
        if line.startswith(("```", "#", "//")):
            continue
        if matches_any(line, PYTHON_CODE_PATTERNS):
            code_like += 1

    return code_like


def has_code_substance(row: pd.Series, text: str) -> bool:
    return (
        to_int(row.get("code_line_count", 0)) >= 2
        or to_int(row.get("code_char_count", 0)) >= 35
        or count_code_like_lines(text) >= 2
        or len(re.findall(r"\b(return|if|for|while|try|except|with|print)\b", text)) >= 2
    )


def compact_code_text(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"```[a-zA-Z0-9_+-]*", "", text)
    text = re.sub(r"[^a-zа-я0-9_]+", "", text, flags=re.IGNORECASE)
    return text


def answer_changes_prompt_code(row: pd.Series, text: str) -> bool:
    """
    Признак для bug_fixing: ответ содержит код и отличается от ошибочного кода в промпте.
    Особенно важен для коротких code-only ответов без слов "исправлено".
    """
    prompt = get_prompt_text(row)
    answer_compact = compact_code_text(text)
    prompt_compact = compact_code_text(prompt)

    if not answer_compact or not prompt_compact:
        return False

    if answer_compact in prompt_compact:
        return False

    return has_code(row, text) and len(answer_compact) >= 12


def count_test_case_lines(text: str) -> int:
    lines = [line.strip() for line in normalize_text(text).split("\n") if line.strip()]
    count = 0

    for line in lines:
        lower = line.lower()
        if re.match(r"^([-*]|\d+[.)])\s+", line):
            if contains_any(lower, ["тест", "кейс", "ожида", "провер", "=>", "->", "true", "false", "none", "ошиб", "исключ"]):
                count += 1
        elif contains_any(lower, ["ожидаемый результат", "expected", "проверка", "тест-кейс"]):
            count += 1

    return count


def has_test_case_format(text: str) -> bool:
    return count_test_case_lines(text) >= 2 or contains_any(text, ["ожидаемый результат", "тест-кейс", "expected result"])


def has_positive_case(text: str) -> bool:
    return contains_any(text, POSITIVE_TEST_WORDS) or bool(re.search(r"\btrue\b|ожида.*true|результат.*true", text, flags=re.IGNORECASE))


def has_negative_case(text: str) -> bool:
    return contains_any(text, NEGATIVE_TEST_WORDS) or bool(re.search(r"\bfalse\b|ожида.*false|ошиб|исключ", text, flags=re.IGNORECASE))


def has_boundary_case(text: str) -> bool:
    return contains_any(text, BOUNDARY_TEST_WORDS)


def length_saturation_penalty(row: pd.Series, score: float) -> float:
    """
    Снижает массовое насыщение оценок до 1.0.
    Короткий ответ может быть хорошим, но максимальный балл дается только при достаточных признаках выполнения задачи.
    """
    length = get_length_level(row)
    word_count = max(to_int(row.get("word_count", 0)), to_int(row.get("text_word_count", 0)))
    expected = get_expected_response_type(row)

    if length == "short" and expected in {"code", "text", "test_cases"}:
        return min(score, 0.92)

    if length in {"medium", "long"} and word_count < 25 and expected not in {"code"}:
        return min(score, 0.85)

    return score


def has_complexity(row: pd.Series, text: str) -> bool:
    return bool_feature(row, "has_complexity") or contains_any(text, COMPLEXITY_PATTERNS) or matches_any(text, [r"\bO\s*\([^)]+\)"])


def has_example(row: pd.Series, text: str) -> bool:
    return bool_feature(row, "has_example") or contains_any(text, EXAMPLE_WORDS)


def has_step_by_step(row: pd.Series, text: str) -> bool:
    return bool_feature(row, "has_step_by_step") or contains_any(text, [
        "сначала", "затем", "далее", "после этого", "на первом шаге",
        "на втором шаге", "1.", "2.", "3.",
    ])


def has_error_explanation(row: pd.Series, text: str) -> bool:
    return bool_feature(row, "has_error_explanation") or contains_any(text, ERROR_WORDS)


def has_fixed_code(row: pd.Series, text: str) -> bool:
    return bool_feature(row, "has_fixed_code") or contains_any(text, FIX_WORDS)


def answer_has_known_bugfix(row: pd.Series, text: str) -> bool:
    """
    Более строгая проверка исправления для текущего набора учебных задач.
    Нужна, чтобы ответ не получал максимум только за слова "ошибка" и наличие кода.
    """
    prompt = get_prompt_text(row).lower()
    answer = normalize_text(text).lower()

    # 1. def add_one(x): return x + '1'
    if "return x + '1'" in prompt or 'return x + "1"' in prompt:
        return bool(re.search(r"return\s+x\s*\+\s*1\b", answer))

    # 2. numbers = [1, 2, 3]; print(numbers[3])
    if "numbers = [1, 2, 3]" in prompt and "numbers[3]" in prompt:
        return (
            "numbers[2]" in answer
            or "numbers[-1]" in answer
            or "len(numbers) - 1" in answer
            or "len(numbers)-1" in answer
        )

    # 3. for i in range(5) без двоеточия
    if "for i in range(5)" in prompt and "print(i)" in prompt:
        return "for i in range(5):" in answer

    # 4. value = int('abc')
    if "value = int('abc')" in prompt or 'value = int("abc")' in prompt:
        if "try:" in answer and "except" in answer:
            return True
        return not ("int('abc')" in answer or 'int("abc")' in answer) and "int(" in answer

    # 5. divide(a, b), деление на ноль
    if "def divide(a, b)" in prompt and "divide(10, 0)" in prompt:
        return (
            "b == 0" in answer
            or "b==0" in answer
            or "zerodivisionerror" in answer
            or "делени" in answer and "ноль" in answer
        )

    # 6. data['b'] при словаре {'a': 1}
    if "data = {'a': 1}" in prompt and "data['b']" in prompt:
        return (
            ".get(" in answer
            or "\"b\"" in answer and ":" in answer
            or "'b'" in answer and ":" in answer
            or "keyerror" in answer
        )

    return False


def answer_keeps_original_bug(row: pd.Series, text: str) -> bool:
    """
    Ищет случаи, когда ответ фактически оставил исходную ошибку.
    Это не стопроцентная проверка, но она хорошо ловит явные ложные максимумы.
    """
    prompt = get_prompt_text(row).lower()
    answer = normalize_text(text).lower()

    bug_fragments = [
        "return x + '1'",
        'return x + "1"',
        "numbers[3]",
        "for i in range(5)\n    print(i)",
        "for i in range(5)\n print(i)",
        "value = int('abc')",
        'value = int("abc")',
        "print(divide(10, 0))",
        "data['b']",
    ]

    for fragment in bug_fragments:
        if fragment in prompt and fragment in answer:
            # Для data['b'] и divide исходный фрагмент может упоминаться в объяснении,
            # поэтому штрафуем только если нет явного исправления.
            if not answer_has_known_bugfix(row, text):
                return True

    return False


def cap_score(score: float, cap: float) -> float:
    return min(score, cap)


def prompt_requires_example(row: pd.Series) -> bool:
    prompt = get_prompt_text(row).lower()
    return contains_any(prompt, ["пример", "покажи", "приведи"])


def prompt_requires_complexity(row: pd.Series) -> bool:
    prompt = get_prompt_text(row).lower()
    return contains_any(prompt, ["сложность", "временную сложность", "o("])


def prompt_requires_positive_negative(row: pd.Series) -> bool:
    prompt = get_prompt_text(row).lower()
    return contains_any(prompt, ["позитив", "негатив"])


def prompt_requires_boundary(row: pd.Series) -> bool:
    prompt = get_prompt_text(row).lower()
    return contains_any(prompt, ["гранич", "краев"])


def score_code_generation(row: pd.Series) -> float:
    text = get_answer_text(row)
    score = 0.0

    if not text or is_error_response(text):
        return 0.0

    code_present = has_code(row, text)
    function_present = has_function_def(row, text)
    code_substantial = has_code_substance(row, text)
    explanation_ok = has_explanation_text(row, text)

    if code_present:
        score += 0.24

    if function_present:
        score += 0.22

    if code_substantial:
        score += 0.18

    if has_relevance(row):
        score += 0.14

    if explanation_required(row):
        if explanation_ok:
            score += 0.10
    else:
        score += 0.06

    if has_example(row, text) or has_step_by_step(row, text):
        score += 0.04

    if code_present or code_substantial:
        score += 0.04

    # Для rule-based оценки максимум лучше оставлять редким.
    # Хорошая генерация кода получает 0.88-0.94, а не автоматически 1.0.
    if code_present and function_present and code_substantial and explanation_ok and has_relevance(row):
        score = cap_score(score, 0.94)
    else:
        score = cap_score(score, 0.88)

    return clamp_score(length_saturation_penalty(row, score))


def score_bug_fixing(row: pd.Series) -> float:
    text = get_answer_text(row)
    score = 0.0

    if not text or is_error_response(text):
        return 0.0

    code_present = has_code(row, text)
    changed_code = answer_changes_prompt_code(row, text)
    known_fix = answer_has_known_bugfix(row, text)
    keeps_bug = answer_keeps_original_bug(row, text)
    error_explained = has_error_explanation(row, text)
    explanation_ok = has_explanation_text(row, text)

    if code_present:
        score += 0.20

    if known_fix:
        score += 0.30
    elif has_fixed_code(row, text) or changed_code:
        score += 0.18

    if error_explained:
        score += 0.16

    if contains_any(text, ERROR_WORDS):
        score += 0.08

    if has_code_substance(row, text) or has_function_def(row, text):
        score += 0.10

    if explanation_required(row):
        if explanation_ok:
            score += 0.10
    else:
        if code_present and (known_fix or changed_code):
            score += 0.08
        else:
            score += 0.04

    if has_relevance(row):
        score += 0.04

    if code_present or error_explained:
        score += 0.02

    # Явно не исправленный код не должен получать высокий балл.
    if keeps_bug:
        score = cap_score(score, 0.55)

    # Code-only исправления могут быть хорошими, но без объяснения не должны уходить в максимум.
    expected = get_expected_response_type(row)
    if expected == "code":
        if known_fix:
            score = cap_score(score, 0.84)
        else:
            score = cap_score(score, 0.72)

    # Для code_and_text максимум даем только при явном исправлении + объяснении.
    if expected == "code_and_text":
        if known_fix and error_explained and explanation_ok:
            score = cap_score(score, 0.96)
        elif known_fix and explanation_ok:
            score = cap_score(score, 0.90)
        else:
            score = cap_score(score, 0.82)

    return clamp_score(length_saturation_penalty(row, score))


def score_algorithm_explanation(row: pd.Series) -> float:
    text = get_answer_text(row)
    score = 0.0

    if not text or is_error_response(text):
        return 0.0

    text_lower = text.lower()
    complexity_required = prompt_requires_complexity(row)
    example_required = prompt_requires_example(row)
    complexity_ok = has_complexity(row, text)
    example_ok = has_example(row, text)

    if contains_any(text_lower, ALGORITHM_WORDS):
        score += 0.24

    if has_step_by_step(row, text):
        score += 0.17

    if complexity_required:
        if complexity_ok:
            score += 0.20
    else:
        score += 0.10 if complexity_ok else 0.04

    if example_required:
        if example_ok:
            score += 0.14
    else:
        score += 0.07 if example_ok else 0.03

    if to_int(row.get("word_count", 0)) >= 20 or to_int(row.get("text_word_count", 0)) >= 20:
        score += 0.10

    if not has_code(row, text) or to_float(row.get("code_ratio", 0.0)) <= 0.4:
        score += 0.06

    if has_relevance(row):
        score += 0.05

    # Для кратких текстовых объяснений не завышаем оценку только за длину/структуру.
    if complexity_required and not complexity_ok:
        score = cap_score(score, 0.78)

    if example_required and not example_ok:
        score = cap_score(score, 0.82)

    if complexity_ok and (example_ok or not example_required) and has_relevance(row):
        score = cap_score(score, 0.94)
    else:
        score = cap_score(score, 0.88)

    return clamp_score(length_saturation_penalty(row, score))


def score_testing(row: pd.Series) -> float:
    text = get_answer_text(row)
    score = 0.0

    if not text or is_error_response(text):
        return 0.0

    expected = get_expected_response_type(row)
    assert_or_framework = has_assert_or_framework(row, text)
    test_case_format = has_test_case_format(text)
    test_case_count = count_test_case_lines(text)

    positive_ok = has_positive_case(text)
    negative_ok = has_negative_case(text)
    boundary_ok = has_boundary_case(text)

    if contains_any(text, TEST_WORDS) or bool_feature(row, "has_assert") or test_case_format:
        score += 0.18

    if expected == "test_cases":
        if test_case_format:
            score += 0.24
        elif assert_or_framework:
            score += 0.16
    else:
        if assert_or_framework:
            score += 0.26
        elif test_case_format:
            score += 0.14

    if positive_ok:
        score += 0.12
    elif not prompt_requires_positive_negative(row) and (assert_or_framework or test_case_format):
        score += 0.05

    if negative_ok:
        score += 0.14
    elif not prompt_requires_positive_negative(row) and (assert_or_framework or test_case_format):
        score += 0.05

    if boundary_ok:
        score += 0.14
    elif not prompt_requires_boundary(row) and (assert_or_framework or test_case_format):
        score += 0.05

    if has_code(row, text) or assert_or_framework:
        score += 0.06

    if explanation_required(row):
        if has_explanation_text(row, text):
            score += 0.05
    else:
        score += 0.03

    if has_relevance(row):
        score += 0.03

    # Разные потолки для исполняемых unit-тестов и текстовых тест-кейсов.
    if expected == "tests":
        if assert_or_framework and positive_ok and negative_ok and boundary_ok:
            score = cap_score(score, 0.96)
        elif assert_or_framework:
            score = cap_score(score, 0.90)
        else:
            score = cap_score(score, 0.82)

    elif expected == "test_cases":
        if test_case_count >= 3 and positive_ok and negative_ok and boundary_ok:
            score = cap_score(score, 0.94)
        elif test_case_count >= 2:
            score = cap_score(score, 0.86)
        else:
            score = cap_score(score, 0.68)

    else:
        score = cap_score(score, 0.90)

    return clamp_score(length_saturation_penalty(row, score))


def calculate_task_score(row: pd.Series) -> float:
    category = normalize_lower(row.get("category", ""))

    if category == "code_generation":
        return score_code_generation(row)

    if category == "bug_fixing":
        return score_bug_fixing(row)

    if category == "algorithm_explanation":
        return score_algorithm_explanation(row)

    if category == "testing":
        return score_testing(row)

    return 0.0


def apply_task_scores(input_path: str | Path, output_path: str | Path) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = {"answer_id", "prompt_id", "category", "answer_text"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"В файле отсутствуют обязательные столбцы: {sorted(missing)}")

    if df.empty:
        print("Файл пустой. Колонка task_score будет создана, строки отсутствуют.")
        df["task_score"] = pd.Series(dtype="float64")
    else:
        df["task_score"] = df.apply(calculate_task_score, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Task score рассчитан: {output_path}")
    print(f"Строк обработано: {len(df)}")

    if not df.empty:
        print("Средний task_score по категориям:")
        print(df.groupby("category")["task_score"].mean().round(4).to_string())

        if "model_name" in df.columns:
            print("\nСредний task_score по моделям:")
            print(df.groupby("model_name")["task_score"].mean().round(4).to_string())

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Расчет task_score для ответов нейросетевых моделей"
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Путь к входному answers.csv",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Путь к выходному answers.csv",
    )

    args = parser.parse_args()

    apply_task_scores(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
