"""
Formal scoring block.

Calculates:
- formal_clarity_score
- formal_structure_score
- formal_utility_score
- formal_relevance_score
- length_match_score
- formal_total_score

Loads weights from configs/scoring_weights.json.
By default overwrites data/processed/answers.csv.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


# Определение путей от корня проекта
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "processed" / "answers.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "answers.csv"
DEFAULT_WEIGHTS_PATH = BASE_DIR / "configs" / "scoring_weights.json"

FORMAL_SCORE_COLUMNS = [
    "formal_clarity_score",
    "formal_structure_score",
    "formal_utility_score",
    "formal_relevance_score",
    "length_match_score",
    "formal_total_score",
]

CODE_TYPES = {"code", "code_and_text", "tests", "test_cases"}


def load_weights(weights_path: Path) -> dict:
    if not weights_path.exists():
        print(f"Внимание: Файл весов не найден по пути {weights_path}. Будут использованы дефолтные веса.")
        return {}
    with open(weights_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("formal_score", {})


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        if value is None or math.isnan(value):
            return low
    except TypeError:
        return low

    return max(low, min(high, value))


def f(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        value = row.get(col, default)
        return default if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default


def s(row: pd.Series, col: str, default: str = "") -> str:
    value = row.get(col, default)
    return default if pd.isna(value) else str(value)


def b(row: pd.Series, col: str) -> bool:
    return bool(f(row, col))


def score_by_ranges(value: float, rules: list[tuple[float, float]] | list[list[float]]) -> float:
    for threshold, score in rules:
        if value >= threshold:
            return float(score)
    return 0.0


def is_error_answer(row: pd.Series) -> bool:
    text = s(row, "answer_text").strip()
    return not text or text.upper().startswith("ERROR")


def is_code_expected(row: pd.Series) -> bool:
    return s(row, "expected_response_type").lower() in CODE_TYPES


def effective_content_size(row: pd.Series) -> float:
    word_count = f(row, "word_count")
    text_word_count = f(row, "text_word_count")
    code_line_count = f(row, "code_line_count")
    char_count = f(row, "char_count")

    size = max(word_count, text_word_count) + code_line_count * 4

    if size == 0 and char_count > 0:
        size = char_count / 8

    return size


def has_minimum_content(row: pd.Series) -> bool:
    if is_error_answer(row):
        return False

    word_count = f(row, "word_count")
    code_line_count = f(row, "code_line_count")
    char_count = f(row, "char_count")

    if is_code_expected(row):
        return code_line_count >= 1 or char_count >= 20 or word_count >= 3

    return word_count >= 5 or char_count >= 30


def length_match(row: pd.Series, weights: dict) -> float:
    """
    Checks whether answer length matches requested length_level:
    short, medium, long.
    """
    if is_error_answer(row):
        return 0.0

    level = s(row, "length_level").lower().strip()
    size = effective_content_size(row)

    if size < 3:
        return 0.0

    w = weights.get("length_match", {})
    code_ranges = w.get("code_expected_ranges", {"short": [8, 110], "medium": [45, 280], "long": [120, 800]})
    text_ranges = w.get("text_expected_ranges", {"short": [15, 100], "medium": [70, 240], "long": [160, 700]})
    ratios = w.get("ratios", [[0.75, 0.8], [0.5, 0.6], [0.25, 0.4], [0.0, 0.2]])

    ranges = code_ranges if is_code_expected(row) else text_ranges

    if level not in ranges:
        return 1.0

    min_size, max_size = ranges[level]

    if min_size <= size <= max_size:
        return 1.0

    ratio = size / min_size if size < min_size else max_size / size

    for limit, score in ratios:
        if ratio >= limit:
            return float(score)

    return 0.2


def category_utility_bonus(row: pd.Series, weights: dict) -> float:
    w = weights.get("utility", {}).get("category_bonuses", {})
    category = s(row, "category").lower()
    expected_type = s(row, "expected_response_type").lower()
    word_count = f(row, "word_count")

    score = 0.0

    if category == "code_generation":
        cw = w.get("code_generation", {})
        score += cw.get("code", 0.15) if b(row, "has_code") else 0
        score += cw.get("func", 0.15) if b(row, "has_function_def") else 0
        score += cw.get("import", 0.05) if b(row, "has_import") else 0
        score += cw.get("code_and_text_len", 0.10) if expected_type == "code_and_text" and word_count >= 20 else 0

    elif category == "bug_fixing":
        cw = w.get("bug_fixing", {})
        score += cw.get("fixed", 0.20) if b(row, "has_fixed_code") else 0
        score += cw.get("error_expl", 0.15) if b(row, "has_error_explanation") else 0
        score += cw.get("code", 0.10) if b(row, "has_code") else 0

    elif category == "algorithm_explanation":
        cw = w.get("algorithm_explanation", {})
        score += cw.get("steps", 0.20) if b(row, "has_step_by_step") else 0
        score += cw.get("complex", 0.15) if b(row, "has_complexity") else 0
        score += cw.get("example", 0.10) if b(row, "has_example") else 0

    elif category == "testing":
        cw = w.get("testing", {})
        score += cw.get("assert", 0.15) if b(row, "has_assert") else 0
        score += cw.get("framework", 0.15) if b(row, "has_pytest") or b(row, "has_unittest") else 0
        score += cw.get("code", 0.10) if b(row, "has_code") else 0
        score += cw.get("example", 0.05) if b(row, "has_example") else 0

    else:
        score += w.get("default", 0.05)

    return score


def category_relevance_bonus(row: pd.Series, weights: dict) -> float:
    w = weights.get("relevance", {})
    category = s(row, "category").lower()

    checks = {
        "code_generation": b(row, "has_code") or b(row, "has_function_def"),
        "bug_fixing": b(row, "has_error_explanation") or b(row, "has_fixed_code"),
        "algorithm_explanation": b(row, "has_complexity") or b(row, "has_step_by_step"),
        "testing": b(row, "has_assert") or b(row, "has_pytest") or b(row, "has_unittest"),
    }

    return w.get("category_bonus", 0.14) if checks.get(category, False) else 0.0


def expected_type_relevance_bonus(row: pd.Series, weights: dict) -> float:
    w = weights.get("relevance", {}).get("expected_type_bonuses", {})
    expected_type = s(row, "expected_response_type").lower()
    answer_text = s(row, "answer_text").lower()

    has_code = b(row, "has_code") or f(row, "code_block_count") >= 1

    if expected_type in {"code", "code_and_text"}:
        return w.get("code_family", 0.25) if has_code else 0.0

    if expected_type in {"tests", "test_cases"}:
        has_test = (
            b(row, "has_assert")
            or b(row, "has_pytest")
            or b(row, "has_unittest")
            or "test" in answer_text
            or "тест" in answer_text
        )
        return w.get("tests_family", 0.25) if has_code or has_test else 0.0

    if expected_type == "structured_text":
        return w.get("structured_text", 0.25) if f(row, "list_count") >= 1 or f(row, "paragraph_count") >= 2 else 0.0

    if expected_type == "text":
        return w.get("text", 0.15)

    return w.get("default", 0.10)


def calculate_formal_clarity_score(row: pd.Series, weights: dict) -> float:
    if is_error_answer(row):
        return 0.0

    w = weights.get("clarity", {})
    score = 0.0
    size = effective_content_size(row)

    score += score_by_ranges(size, w.get("size_ranges", [[30, 0.20], [10, 0.12], [5, 0.05]]))
    score += score_by_ranges(f(row, "sentence_count"), w.get("sentence_count_ranges", [[3, 0.20], [1, 0.10]]))

    avg_len = f(row, "avg_sentence_length")
    avg_len_w = w.get("avg_len", {})
    
    opt = avg_len_w.get("optimal", [8, 28, 0.25])
    acc = avg_len_w.get("acceptable", [5, 40, 0.15])
    min_len = avg_len_w.get("min", [0, 999, 0.05])

    if opt[0] <= avg_len <= opt[1]:
        score += opt[2]
    elif acc[0] <= avg_len <= acc[1]:
        score += acc[2]
    elif avg_len > min_len[0]:
        score += min_len[2]

    score += score_by_ranges(
        f(row, "readability_score"),
        w.get("readability_ranges", [[0.7, 0.20], [0.4, 0.10], [0.001, 0.05]]),
    )

    lexical = f(row, "lexical_diversity")
    lexical_w = w.get("lexical", {})
    lex_opt = lexical_w.get("optimal", [0.25, 0.85, 0.10])
    lex_min = lexical_w.get("min", [0, 999, 0.05])

    if lex_opt[0] <= lexical <= lex_opt[1]:
        score += lex_opt[2]
    elif lexical > lex_min[0]:
        score += lex_min[2]

    if (
        s(row, "expected_response_type").lower() == "code"
        and b(row, "has_code")
        and f(row, "word_count") == 0
        and f(row, "code_line_count") >= 1
    ):
        score += w.get("code_no_words_bonus", 0.18)

    score += w.get("length_match_bonus", 0.05) * length_match(row, weights)

    return round(clamp(score), 4)


def calculate_formal_structure_score(row: pd.Series, weights: dict) -> float:
    if is_error_answer(row):
        return 0.0

    w = weights.get("structure", {})
    score = 0.0

    score += score_by_ranges(f(row, "paragraph_count"), w.get("paragraph_ranges", [[2, 0.20], [1, 0.10]]))
    score += score_by_ranges(f(row, "list_count"), w.get("list_ranges", [[2, 0.20], [1, 0.15]]))
    score += score_by_ranges(f(row, "sentence_count"), w.get("sentence_ranges", [[3, 0.15], [1, 0.08]]))

    if f(row, "structure_score") > 0:
        score += w.get("structure_score_multiplier", 0.25) * clamp(f(row, "structure_score"))

    expected_type = s(row, "expected_response_type").lower()

    if expected_type in CODE_TYPES:
        cw = w.get("code_expected", {})
        score += cw.get("block_bonus", 0.20) if f(row, "code_block_count") >= 1 else 0
        score += score_by_ranges(
            f(row, "code_line_count"),
            cw.get("line_ranges", [[3, 0.12], [2, 0.10], [1, 0.05]]),
        )

    elif expected_type == "structured_text":
        score += w.get("structured_text_bonus", 0.20) if f(row, "paragraph_count") >= 2 or f(row, "list_count") >= 1 else 0

    else:
        score += w.get("text_bonus", 0.10) if f(row, "paragraph_count") >= 1 else 0

    return round(clamp(score), 4)


def calculate_formal_utility_score(row: pd.Series, weights: dict) -> float:
    if is_error_answer(row):
        return 0.0

    w = weights.get("utility", {})
    score = 0.0
    size = effective_content_size(row)

    score += score_by_ranges(size, w.get("size_ranges", [[40, 0.12], [15, 0.08], [5, 0.03]]))
    score += score_by_ranges(f(row, "technical_terms_count"), w.get("terms_ranges", [[5, 0.15], [1, 0.08]]))
    score += w.get("list_bonus", 0.10) if f(row, "list_count") >= 1 else 0

    if is_code_expected(row):
        score += w.get("code_expected_bonus", 0.20) if b(row, "has_code") or f(row, "code_block_count") >= 1 else 0
    else:
        score += w.get("text_expected_bonus", 0.10)

    score += category_utility_bonus(row, weights)
    score += w.get("length_match_bonus", 0.08) * length_match(row, weights)

    return round(clamp(score), 4)


def calculate_formal_relevance_score(row: pd.Series, weights: dict) -> float:
    if is_error_answer(row):
        return 0.0

    w = weights.get("relevance", {})
    score = 0.0
    size = effective_content_size(row)

    overlap = max(f(row, "prompt_text_overlap"), f(row, "prompt_full_overlap"))

    score += score_by_ranges(overlap, w.get("overlap_ranges", [[0.5, 0.35], [0.25, 0.25], [0.001, 0.12]]))
    score += score_by_ranges(size, w.get("size_ranges", [[20, 0.18], [5, 0.08], [0.001, 0.03]]))

    score += expected_type_relevance_bonus(row, weights)
    score += category_relevance_bonus(row, weights)
    score += w.get("length_match_bonus", 0.08) * length_match(row, weights)

    return round(clamp(score), 4)


def calculate_formal_total_score(row: pd.Series, weights: dict) -> float:
    total_weights = weights.get("total_weights", {
        "formal_clarity_score": 0.22,
        "formal_structure_score": 0.22,
        "formal_utility_score": 0.22,
        "formal_relevance_score": 0.22,
        "length_match_score": 0.12,
    })
    
    total = sum(f(row, col) * weight for col, weight in total_weights.items())
    return round(clamp(total), 4)


def validate_input_columns(df: pd.DataFrame) -> None:
    required = {
        "answer_text",
        "category",
        "length_level",
        "expected_response_type",
        "word_count",
        "text_word_count",
        "sentence_count",
        "paragraph_count",
        "avg_sentence_length",
        "list_count",
        "has_code",
        "code_block_count",
        "code_line_count",
        "char_count",
        "technical_terms_count",
        "readability_score",
        "lexical_diversity",
        "structure_score",
        "prompt_text_overlap",
        "prompt_full_overlap",
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
    }

    missing = sorted(required - set(df.columns))

    if missing:
        print("Warning: missing columns:")
        for col in missing:
            print(f"  - {col}")
        print("Missing columns will be treated as zero/empty values.\n")


def print_examples(df: pd.DataFrame, title: str, mask: pd.Series, limit: int = 10) -> None:
    subset = df[mask]

    print(f"{title}: {len(subset)}")

    if subset.empty:
        print()
        return

    cols = [
        "answer_id",
        "model_name",
        "category",
        "length_level",
        "expected_response_type",
        "word_count",
        "text_word_count",
        "code_line_count",
        "char_count",
        "_effective_content_size",
        "length_match_score",
        "formal_total_score",
    ]

    existing = [col for col in cols if col in df.columns]

    print()
    print(subset[existing].head(limit).to_string(index=False))
    print()


def run_quality_checks(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("Quality checks")
    print("=" * 80)

    df["_effective_content_size"] = df.apply(effective_content_size, axis=1)
    df["_has_minimum_content"] = df.apply(has_minimum_content, axis=1)

    print_examples(
        df,
        "Rows without minimum content",
        df["_has_minimum_content"] == False,
    )

    print_examples(
        df,
        "Rows with formal_total_score = 0",
        df["formal_total_score"] == 0,
    )

    if "length_level" in df.columns:
        print_examples(
            df,
            "Short answers with formal_total_score < 0.4",
            (df["length_level"] == "short") & (df["formal_total_score"] < 0.4),
        )

        print_examples(
            df,
            "Short answers with effective_content_size > 120",
            (df["length_level"] == "short") & (df["_effective_content_size"] > 120),
        )

        print_examples(
            df,
            "Low-scored long answers with effective_content_size < 100",
            (df["length_level"] == "long")
            & (df["_effective_content_size"] < 100)
            & (df["formal_total_score"] < 0.6),
        )

        print("Length match distribution by length_level:")
        print(df.groupby("length_level")["length_match_score"].describe().round(4))

    print("=" * 80 + "\n")


def apply_formal_scores(input_path: str | Path, output_path: str | Path, weights_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)
    weights_path = Path(weights_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    weights = load_weights(weights_path)
    df = pd.read_csv(input_path)

    validate_input_columns(df)

    score_functions = {
        "formal_clarity_score": calculate_formal_clarity_score,
        "formal_structure_score": calculate_formal_structure_score,
        "formal_utility_score": calculate_formal_utility_score,
        "formal_relevance_score": calculate_formal_relevance_score,
        "length_match_score": length_match,
    }

    # Итерация с передачей словаря weights через lambda
    for column, func in score_functions.items():
        df[column] = df.apply(lambda row: func(row, weights), axis=1)

    df["formal_total_score"] = df.apply(lambda row: calculate_formal_total_score(row, weights), axis=1)

    run_quality_checks(df)

    df = df.drop(
        columns=["_effective_content_size", "_has_minimum_content"],
        errors="ignore",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def print_summary(df: pd.DataFrame, output_path: str | Path) -> None:
    print(f"Formal scores calculated for {len(df)} answers.")
    print(f"Saved to: {output_path}\n")

    print("Score distribution:")
    print(df[FORMAL_SCORE_COLUMNS].describe().round(4))
    print()

    if {"model_name", "category"}.issubset(df.columns):
        print("Mean scores by model and category:")
        print(df.groupby(["model_name", "category"])[FORMAL_SCORE_COLUMNS].mean().round(4))
        print()

    if {"model_name", "length_level"}.issubset(df.columns):
        print("Mean scores by model and length_level:")
        print(
            df.groupby(["model_name", "length_level"])[
                ["length_match_score", "formal_total_score"]
            ]
            .mean()
            .round(4)
        )
        print()

    if "answer_text" in df.columns:
        text = df["answer_text"].fillna("").astype(str)
        print(f"Error answers: {text.str.upper().str.startswith('ERROR').sum()}")
        print(f"Empty answers: {text.str.strip().eq('').sum()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate formal scores for model answers.")

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to processed answers CSV.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Path to output CSV. If not specified, input file will be overwritten.",
    )
    
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Path to JSON weights file (scoring_weights.json)",
    )

    args = parser.parse_args()

    output_path = args.output or args.input

    df = apply_formal_scores(args.input, output_path, args.weights)
    print_summary(df, output_path)


if __name__ == "__main__":
    main()