# src/validation/combine_answers.py

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


MODEL_1_PATH = Path("data/raw/answers_model_1_raw.csv")
MODEL_2_PATH = Path("data/raw/answers_model_2_raw.csv")
OUTPUT_PATH = Path("data/raw/answers_raw.csv")


PREFERRED_ORDER = [
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
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def validate_answers_file(df: pd.DataFrame, file_label: str) -> None:
    required_columns = {
        "answer_id",
        "prompt_id",
        "model_name",
        "model_version",
        "answer_text",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{file_label}: отсутствуют обязательные столбцы: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"{file_label}: файл пустой")

    duplicated = df.duplicated(
        subset=["prompt_id", "model_name", "model_version"],
        keep=False,
    )

    if duplicated.any():
        examples = (
            df.loc[duplicated, ["prompt_id", "model_name", "model_version"]]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            f"{file_label}: найдены дубли prompt_id + model_name + model_version. "
            f"Примеры: {examples}"
        )


def combine_answers() -> Path:
    df1 = read_csv(MODEL_1_PATH)
    df2 = read_csv(MODEL_2_PATH)

    validate_answers_file(df1, "Файл модели 1")
    validate_answers_file(df2, "Файл модели 2")

    combined_df = pd.concat([df1, df2], ignore_index=True)

    duplicated = combined_df.duplicated(
        subset=["prompt_id", "model_name", "model_version"],
        keep=False,
    )

    if duplicated.any():
        examples = (
            combined_df.loc[duplicated, ["prompt_id", "model_name", "model_version"]]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "В объединенном файле найдены дубли "
            "prompt_id + model_name + model_version. "
            f"Примеры: {examples}"
        )

    existing_columns = [
        column for column in PREFERRED_ORDER
        if column in combined_df.columns
    ]

    remaining_columns = [
        column for column in combined_df.columns
        if column not in existing_columns
    ]

    combined_df = combined_df[existing_columns + remaining_columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=== Объединение ответов ===")
    print(f"Модель 1: {MODEL_1_PATH} — {len(df1)} строк")
    print(f"Модель 2: {MODEL_2_PATH} — {len(df2)} строк")
    print(f"Итого: {len(combined_df)} строк")
    print(f"Сохранено: {OUTPUT_PATH}")

    return OUTPUT_PATH


def main() -> int:
    try:
        combine_answers()
    except Exception as error:
        print(f"[ERROR] {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())