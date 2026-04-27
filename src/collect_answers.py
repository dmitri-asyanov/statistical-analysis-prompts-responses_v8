import argparse
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.models.base_client import BaseModelClient
from src.models.yandex_client import YandexGPTClient
from src.models.second_model_client import GigaChatClient


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

PROMPTS_RAW_PATH = RAW_DATA_DIR / "prompts_raw.csv"

ANSWERS_MODEL_1_RAW_PATH = RAW_DATA_DIR / "answers_model_1_raw.csv"
ANSWERS_MODEL_2_RAW_PATH = RAW_DATA_DIR / "answers_model_2_raw.csv"
ANSWERS_RAW_PATH = RAW_DATA_DIR / "answers_raw.csv"


PROMPT_COLUMNS_TO_ADD = [
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
]

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


def get_client(model_key: str) -> BaseModelClient:
    model_key = model_key.strip().lower()

    if model_key in {"yandex", "yandexgpt", "yandexgpt-lite"}:
        return YandexGPTClient()

    if model_key in {"gigachat", "giga", "second"}:
        return GigaChatClient()

    raise ValueError(f"Неизвестная модель: {model_key}")


def load_prompts(prompts_raw_path: Path) -> pd.DataFrame:
    if not prompts_raw_path.exists():
        raise FileNotFoundError(f"Файл не найден: {prompts_raw_path}")

    prompts_df = pd.read_csv(prompts_raw_path)

    if prompts_df.empty:
        raise ValueError("prompts_raw.csv пустой")

    required_columns = {"prompt_id", "prompt_text"}
    missing = required_columns - set(prompts_df.columns)

    if missing:
        raise ValueError(f"В prompts_raw.csv отсутствуют столбцы: {missing}")

    if prompts_df["prompt_id"].duplicated().any():
        duplicated_ids = prompts_df.loc[
            prompts_df["prompt_id"].duplicated(), "prompt_id"
        ].tolist()

        raise ValueError(
            f"В prompts_raw.csv найдены дубли prompt_id. "
            f"Примеры: {duplicated_ids[:10]}"
        )

    return prompts_df


def collect_answers_for_model(
    prompts_raw_path: Path,
    output_path: Path,
    client: BaseModelClient,
) -> Path:
    prompts_df = load_prompts(prompts_raw_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    answers = []

    total = len(prompts_df)

    for index, (_, row) in enumerate(prompts_df.iterrows(), start=1):
        prompt_id = row["prompt_id"]
        prompt_text = row["prompt_text"]

        role_text = (
            row["role_text"]
            if "role_text" in prompts_df.columns and pd.notna(row["role_text"])
            else None
        )

        expected_response_type = (
            row["expected_response_type"]
            if "expected_response_type" in prompts_df.columns and pd.notna(row["expected_response_type"])
            else None
        )

        length_level = (
            row["length_level"]
            if "length_level" in prompts_df.columns and pd.notna(row["length_level"])
            else None
        )

        try:
            start_time = time.time()

            answer_text = client.generate_answer(
                prompt_text=prompt_text,
                role_text=role_text,
                expected_response_type=expected_response_type,
                length_level=length_level,
            )

            response_time = round(time.time() - start_time, 4)

            answers.append({
                "answer_id": str(uuid.uuid4()),
                "prompt_id": prompt_id,
                "model_name": client.model_name,
                "model_version": client.model_version,
                "answer_text": answer_text,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "response_time_sec": response_time,
            })

            print(f"[OK] {index}/{total} {client.model_name} {prompt_id}")

        except Exception as error:
            print(f"[ERROR] {index}/{total} {client.model_name} {prompt_id}: {error}")

            answers.append({
                "answer_id": str(uuid.uuid4()),
                "prompt_id": prompt_id,
                "model_name": client.model_name,
                "model_version": client.model_version,
                "answer_text": f"ERROR: {error}",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "response_time_sec": None,
            })

    answers_df = pd.DataFrame(answers)

    existing_prompt_columns = [
        col for col in PROMPT_COLUMNS_TO_ADD if col in prompts_df.columns
    ]

    answers_df = answers_df.merge(
        prompts_df[existing_prompt_columns],
        on="prompt_id",
        how="left",
        validate="many_to_one",
    )

    existing_columns = [col for col in PREFERRED_ORDER if col in answers_df.columns]
    remaining_columns = [col for col in answers_df.columns if col not in existing_columns]

    answers_df = answers_df[existing_columns + remaining_columns]

    answers_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Сохранено {len(answers_df)} ответов: {output_path}")

    return output_path


def combine_answer_files(
    input_paths: list[Path],
    output_path: Path,
) -> Path:
    frames = []

    for path in input_paths:
        if not path.exists():
            print(f"[SKIP] Файл не найден: {path}")
            continue

        df = pd.read_csv(path)

        if df.empty:
            print(f"[SKIP] Файл пустой: {path}")
            continue

        frames.append(df)

    if not frames:
        raise ValueError("Нет файлов для объединения")

    combined_df = pd.concat(frames, ignore_index=True)

    duplicate_mask = combined_df.duplicated(
        subset=["prompt_id", "model_name", "model_version"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = combined_df.loc[
            duplicate_mask,
            ["prompt_id", "model_name", "model_version"]
        ].head(20)

        raise ValueError(
            "Найдены дубли по prompt_id + model_name + model_version:\n"
            f"{duplicates}"
        )

    existing_columns = [col for col in PREFERRED_ORDER if col in combined_df.columns]
    remaining_columns = [col for col in combined_df.columns if col not in existing_columns]

    combined_df = combined_df[existing_columns + remaining_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Объединенный файл сохранен: {output_path}")
    print(f"Всего строк: {len(combined_df)}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сбор ответов нейросетевых моделей для дипломного проекта"
    )

    parser.add_argument(
        "--model",
        choices=["yandex", "gigachat", "all"],
        required=True,
        help="Какую модель собрать",
    )

    parser.add_argument(
        "--prompts",
        default=str(PROMPTS_RAW_PATH),
        help="Путь к prompts_raw.csv",
    )

    parser.add_argument(
        "--combine",
        action="store_true",
        help="После сбора объединить answers_model_1_raw.csv и answers_model_2_raw.csv",
    )

    args = parser.parse_args()

    prompts_path = Path(args.prompts)

    if args.model in {"yandex", "all"}:
        yandex_client = get_client("yandex")
        collect_answers_for_model(
            prompts_raw_path=prompts_path,
            output_path=ANSWERS_MODEL_1_RAW_PATH,
            client=yandex_client,
        )

    if args.model in {"gigachat", "all"}:
        gigachat_client = get_client("gigachat")
        collect_answers_for_model(
            prompts_raw_path=prompts_path,
            output_path=ANSWERS_MODEL_2_RAW_PATH,
            client=gigachat_client,
        )

    if args.combine or args.model == "all":
        combine_answer_files(
            input_paths=[
                ANSWERS_MODEL_1_RAW_PATH,
                ANSWERS_MODEL_2_RAW_PATH,
            ],
            output_path=ANSWERS_RAW_PATH,
        )


if __name__ == "__main__":
    main()