# src/validation/check_answers.py

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "answer_id",
    "prompt_id",
    "model_name",
    "model_version",
    "answer_text",
}


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def normalize_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def is_empty_text(series: pd.Series) -> pd.Series:
    return normalize_series(series).eq("")


def contains_error(series: pd.Series) -> pd.Series:
    text = normalize_series(series)

    return text.str.match(
        r"^\s*(ERROR:|\[ERROR\]|\[CRITICAL\]|CRITICAL:|Traceback \(most recent call last\))",
        case=False,
    )


def format_examples(values: Iterable[object], limit: int = 10) -> str:
    items = [str(v) for v in values]

    if not items:
        return "нет"

    shown = items[:limit]
    suffix = "" if len(items) <= limit else f" ... и ещё {len(items) - limit}"

    return ", ".join(shown) + suffix


def check_required_columns(
    df: pd.DataFrame,
    file_label: str,
    result: ValidationResult,
) -> bool:
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        result.add_error(
            f"{file_label}: отсутствуют обязательные столбцы: {sorted(missing)}"
        )
        return False

    return True


def check_single_file(
    df: pd.DataFrame,
    file_label: str,
    result: ValidationResult,
    expected_model_name: str | None = None,
    expected_model_version: str | None = None,
) -> None:
    if not check_required_columns(df, file_label, result):
        return

    if df.empty:
        result.add_error(f"{file_label}: файл не содержит строк с ответами")
        return

    for column in ["answer_id", "prompt_id"]:
        empty_mask = is_empty_text(df[column])
        if empty_mask.any():
            result.add_error(
                f"{file_label}: пустые значения в {column}: {int(empty_mask.sum())}"
            )

    empty_answer_mask = is_empty_text(df["answer_text"])
    if empty_answer_mask.any():
        examples = df.loc[empty_answer_mask, "prompt_id"].head(10).tolist()
        result.add_error(
            f"{file_label}: пустые answer_text: {int(empty_answer_mask.sum())}. "
            f"Примеры prompt_id: {format_examples(examples)}"
        )

    error_mask = contains_error(df["answer_text"])
    if error_mask.any():
        examples = df.loc[error_mask, "prompt_id"].head(10).tolist()
        result.add_error(
            f"{file_label}: найдены ERROR/исключения в answer_text: "
            f"{int(error_mask.sum())}. "
            f"Примеры prompt_id: {format_examples(examples)}"
        )

    empty_model_name = is_empty_text(df["model_name"])
    empty_model_version = is_empty_text(df["model_version"])

    if empty_model_name.any():
        result.add_error(
            f"{file_label}: пустые model_name: {int(empty_model_name.sum())}"
        )

    if empty_model_version.any():
        result.add_error(
            f"{file_label}: пустые model_version: {int(empty_model_version.sum())}"
        )

    model_names = sorted(set(normalize_series(df["model_name"])) - {""})
    model_versions = sorted(set(normalize_series(df["model_version"])) - {""})

    if len(model_names) != 1:
        result.add_error(
            f"{file_label}: ожидалось одно значение model_name, найдено: {model_names}"
        )

    if len(model_versions) != 1:
        result.add_error(
            f"{file_label}: ожидалось одно значение model_version, найдено: {model_versions}"
        )

    if expected_model_name is not None:
        wrong_mask = normalize_series(df["model_name"]).ne(expected_model_name)

        if wrong_mask.any():
            result.add_error(
                f"{file_label}: model_name должен быть '{expected_model_name}', "
                f"но отличается в строках: {int(wrong_mask.sum())}"
            )

    if expected_model_version is not None:
        wrong_mask = normalize_series(df["model_version"]).ne(expected_model_version)

        if wrong_mask.any():
            result.add_error(
                f"{file_label}: model_version должен быть '{expected_model_version}', "
                f"но отличается в строках: {int(wrong_mask.sum())}"
            )

    duplicated_prompt_model = df.duplicated(
        subset=["prompt_id", "model_name"],
        keep=False,
    )

    if duplicated_prompt_model.any():
        examples = (
            df.loc[duplicated_prompt_model, ["prompt_id", "model_name"]]
            .drop_duplicates()
            .head(10)
            .apply(lambda row: f"{row['prompt_id']} + {row['model_name']}", axis=1)
            .tolist()
        )

        result.add_error(
            f"{file_label}: дубли prompt_id + model_name: "
            f"{int(duplicated_prompt_model.sum())} строк. "
            f"Примеры: {format_examples(examples)}"
        )


def compare_prompt_sets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    result: ValidationResult,
) -> None:
    if "prompt_id" not in df1.columns or "prompt_id" not in df2.columns:
        return

    prompts_1 = set(normalize_series(df1["prompt_id"])) - {""}
    prompts_2 = set(normalize_series(df2["prompt_id"])) - {""}

    missing_in_model_2 = sorted(prompts_1 - prompts_2)
    extra_in_model_2 = sorted(prompts_2 - prompts_1)

    if missing_in_model_2:
        result.add_error(
            f"Во второй модели отсутствуют prompt_id из первой модели: "
            f"{len(missing_in_model_2)}. "
            f"Примеры: {format_examples(missing_in_model_2)}"
        )

    if extra_in_model_2:
        result.add_error(
            f"Во второй модели есть лишние prompt_id, которых нет в первой модели: "
            f"{len(extra_in_model_2)}. "
            f"Примеры: {format_examples(extra_in_model_2)}"
        )


def check_combined_duplicates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    result: ValidationResult,
) -> None:
    if not {"prompt_id", "model_name"}.issubset(df1.columns):
        return

    if not {"prompt_id", "model_name"}.issubset(df2.columns):
        return

    combined = pd.concat(
        [
            df1.assign(_source="model_1"),
            df2.assign(_source="model_2"),
        ],
        ignore_index=True,
    )

    duplicated = combined.duplicated(
        subset=["prompt_id", "model_name"],
        keep=False,
    )

    if duplicated.any():
        examples = (
            combined.loc[duplicated, ["prompt_id", "model_name", "_source"]]
            .head(20)
            .apply(
                lambda row: (
                    f"{row['prompt_id']} + {row['model_name']} "
                    f"({row['_source']})"
                ),
                axis=1,
            )
            .tolist()
        )

        result.add_error(
            "В объединенном наборе найдены дубли prompt_id + model_name: "
            f"{int(duplicated.sum())} строк. "
            f"Примеры: {format_examples(examples)}"
        )


def print_summary(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    path1: Path,
    path2: Path,
) -> None:
    print("\n=== Сводка ===")
    print(f"Файл модели 1: {path1}")
    print(f"Количество строк модели 1: {len(df1)}")
    print(f"Файл модели 2: {path2}")
    print(f"Количество строк модели 2: {len(df2)}")

    if "prompt_id" in df1.columns and "prompt_id" in df2.columns:
        prompts_1 = set(normalize_series(df1["prompt_id"])) - {""}
        prompts_2 = set(normalize_series(df2["prompt_id"])) - {""}

        print(f"Уникальных prompt_id модели 1: {len(prompts_1)}")
        print(f"Уникальных prompt_id модели 2: {len(prompts_2)}")
        print(f"Общих prompt_id: {len(prompts_1 & prompts_2)}")

    for label, df in [("Модель 1", df1), ("Модель 2", df2)]:
        if {"model_name", "model_version"}.issubset(df.columns):
            names = sorted(set(normalize_series(df["model_name"])) - {""})
            versions = sorted(set(normalize_series(df["model_version"])) - {""})

            print(f"{label}: model_name = {names}")
            print(f"{label}: model_version = {versions}")


def validate(args: argparse.Namespace) -> ValidationResult:
    result = ValidationResult(errors=[], warnings=[])

    path1 = Path(args.model1)
    path2 = Path(args.model2)

    df1 = read_csv(path1)
    df2 = read_csv(path2)

    print_summary(df1, df2, path1, path2)

    if len(df1) != len(df2):
        result.add_error(
            f"Количество строк не совпадает: "
            f"модель 1 = {len(df1)}, модель 2 = {len(df2)}"
        )

    check_single_file(
        df=df1,
        file_label="Файл модели 1",
        result=result,
        expected_model_name=args.model1_name,
        expected_model_version=args.model1_version,
    )

    check_single_file(
        df=df2,
        file_label="Файл модели 2",
        result=result,
        expected_model_name=args.model2_name,
        expected_model_version=args.model2_version,
    )

    compare_prompt_sets(df1, df2, result)
    check_combined_duplicates(df1, df2, result)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Проверка сопоставимости CSV с ответами двух нейросетевых моделей."
    )

    parser.add_argument(
        "--model1",
        default="data/raw/answers_model_1_raw.csv",
        help="Путь к CSV с ответами первой модели.",
    )

    parser.add_argument(
        "--model2",
        default="data/raw/answers_model_2_raw.csv",
        help="Путь к CSV с ответами второй модели.",
    )

    parser.add_argument(
        "--model1-name",
        default=None,
        help="Ожидаемое значение model_name для первой модели. Например: yandexgpt-lite",
    )

    parser.add_argument(
        "--model2-name",
        default=None,
        help="Ожидаемое значение model_name для второй модели. Например: gigachat",
    )

    parser.add_argument(
        "--model1-version",
        default=None,
        help="Ожидаемое значение model_version для первой модели. Например: latest",
    )

    parser.add_argument(
        "--model2-version",
        default=None,
        help="Ожидаемое значение model_version для второй модели. Например: GigaChat",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = validate(args)
    except Exception as exc:
        print(f"\n[CRITICAL] {exc}")
        return 2

    print("\n=== Результат проверки ===")

    if result.warnings:
        print("\nПредупреждения:")
        for warning in result.warnings:
            print(f"[WARNING] {warning}")

    if result.errors:
        print("\nОшибки:")
        for error in result.errors:
            print(f"[ERROR] {error}")

        print("\nИтог: проверка не пройдена")
        return 1

    print("[OK] Проверка пройдена. Данные двух моделей сопоставимы.")
    return 0


if __name__ == "__main__":
    sys.exit(main())