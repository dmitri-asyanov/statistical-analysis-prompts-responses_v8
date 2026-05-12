"""
Judge scoring with DeepSeek API.

Produces:
- judge_relevance
- judge_completeness
- judge_clarity
- judge_logic
- judge_utility
- judge_total_score
- judge_comment

Loads weights from configs/scoring_weights.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# Определение путей от корня проекта
BASE_DIR = Path(__file__).resolve().parents[2]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-v4-pro"
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "processed" / "answers.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "answers.csv"
DEFAULT_WEIGHTS_PATH = BASE_DIR / "configs" / "scoring_weights.json"

JUDGE_SCORE_COLUMNS = [
    "judge_relevance",
    "judge_completeness",
    "judge_clarity",
    "judge_logic",
    "judge_utility",
    "judge_comment",
    "judge_total_score",
]

PARTIAL_SCORE_KEYS = [
    "judge_relevance",
    "judge_completeness",
    "judge_clarity",
    "judge_logic",
    "judge_utility",
]

SYSTEM_PROMPT = """
You are an expert evaluator of answers from neural network models.
Return only valid JSON.
Do not write Markdown.
Do not write explanations outside JSON.
Use numeric values from 0 to 10 for all score fields.
The JSON object must contain exactly these keys:
judge_relevance, judge_completeness, judge_clarity, judge_logic,
judge_utility, judge_comment.
""".strip()

CATEGORY_CRITERIA = {
    "code_generation": """
For code_generation evaluate whether:
- the answer contains a working solution;
- the answer contains code;
- the code matches the task;
- the answer can be used in practice.
""".strip(),
    "bug_fixing": """
For bug_fixing evaluate whether:
- the bug is found;
- a fix is proposed;
- the reason for the bug is explained;
- there is corrected code or a clear instruction.
""".strip(),
    "algorithm_explanation": """
For algorithm_explanation evaluate whether:
- the core idea of the algorithm is explained;
- the explanation has a logical sequence;
- the explanation is clear;
- there is an example or useful interpretation.
""".strip(),
    "testing": """
For testing evaluate whether:
- tests are proposed;
- basic cases are covered;
- edge cases are covered;
- the tests can be used in practice.
""".strip(),
}


def load_weights(weights_path: Path) -> dict:
    if not weights_path.exists():
        print(f"Внимание: Файл весов не найден по пути {weights_path}. Будут использованы дефолтные веса.")
        return {}
    with open(weights_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("judge_score", {})


def build_judge_prompt(prompt_text: str, answer_text: str, category: str) -> str:
    """
    Build a category-aware prompt for judge evaluation.
    """
    normalized_category = str(category or "").strip().lower()
    category_criteria = CATEGORY_CRITERIA.get(
        normalized_category,
        "Evaluate the answer according to the task category and the general criteria.",
    )

    return f"""
Evaluate the model answer for the original prompt.

Task category: {normalized_category or "unknown"}

Category-specific criteria:
{category_criteria}

General criteria:
- judge_relevance: how well the answer matches the prompt;
- judge_completeness: how fully the task is solved;
- judge_clarity: how clear and well-formulated the answer is;
- judge_logic: how logical and internally consistent the answer is;
- judge_utility: how practically useful the answer is.

Return only this JSON object:
{{
  "judge_relevance": number from 0 to 10,
  "judge_completeness": number from 0 to 10,
  "judge_clarity": number from 0 to 10,
  "judge_logic": number from 0 to 10,
  "judge_utility": number from 0 to 10,
  "judge_comment": "short comment explaining the score"
}}

Original prompt:
{prompt_text}

Model answer:
{answer_text}
""".strip()


def _safe_failed_result(reason: str) -> dict[str, Any]:
    return {
        "judge_relevance": None,
        "judge_completeness": None,
        "judge_clarity": None,
        "judge_logic": None,
        "judge_utility": None,
        "judge_comment": f"Judge evaluation failed: {reason}",
        "judge_total_score": None,
    }


def _get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Set it in the environment before running judge scoring."
        )

    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def _strip_json_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def _parse_json_response(content: str) -> dict[str, Any]:
    data = json.loads(_strip_json_fences(content))

    if not isinstance(data, dict):
        raise ValueError("response JSON is not an object")

    return data


def _as_score(value: Any, field_name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is not numeric") from exc

    if not 0 <= score <= 10:
        raise ValueError(f"{field_name} is outside 0..10")

    return round(score, 4)


def _normalize_judge_result(data: dict[str, Any], weights: dict) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key in PARTIAL_SCORE_KEYS:
        result[key] = _as_score(data.get(key), key)

    # Используем базовый вес для всех (0.2 по умолчанию), либо индивидуальные веса
    default_comp_weight = weights.get("components_weight", 0.2)
    
    total_score = 0.0
    for key in PARTIAL_SCORE_KEYS:
        # Если в JSON прописан специфический вес (например, "judge_logic": 0.3), берем его. Иначе дефолтный 0.2.
        weight = weights.get(key, default_comp_weight)
        total_score += result[key] * weight

    result["judge_total_score"] = round(total_score, 4)
    result["judge_comment"] = str(data.get("judge_comment", "")).strip()

    return result


def _request_judge_evaluation(
    client: OpenAI,
    judge_prompt: str,
    model_name: str,
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("empty response from DeepSeek API")

    return content


def get_judge_score(
    prompt_text: str,
    answer_text: str,
    category: str,
    model_name: str | None = None,
    weights: dict | None = None,
) -> dict[str, Any]:
    """
    Evaluate one model answer with DeepSeek and return judge score fields.
    """
    if weights is None:
        weights = {}
        
    client = _get_client()
    selected_model = model_name or DEFAULT_MODEL_NAME
    judge_prompt = build_judge_prompt(prompt_text, answer_text, category)
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            content = _request_judge_evaluation(client, judge_prompt, selected_model)
            data = _parse_json_response(content)
            return _normalize_judge_result(data, weights)
        except json.JSONDecodeError as exc:
            last_error = exc

            if attempt == 0:
                continue

        except Exception as exc:
            last_error = exc
            break

    reason = str(last_error) if last_error else "unknown error"
    return _safe_failed_result(reason)


def score_dataframe_with_judge(
    df: pd.DataFrame,
    weights: dict,
    prompt_col: str = "prompt_text",
    answer_col: str = "answer_text",
    category_col: str = "category",
) -> pd.DataFrame:
    """
    Add judge score columns to a DataFrame.
    """
    missing = [col for col in [prompt_col, answer_col, category_col] if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns for judge scoring: {missing}")

    scored_df = df.copy()
    rows: list[dict[str, Any]] = []

    total_rows = len(scored_df)

    for position, (index, row) in enumerate(scored_df.iterrows(), start=1):
        result = get_judge_score(
            prompt_text="" if pd.isna(row[prompt_col]) else str(row[prompt_col]),
            answer_text="" if pd.isna(row[answer_col]) else str(row[answer_col]),
            category="" if pd.isna(row[category_col]) else str(row[category_col]),
            weights=weights,
        )
        rows.append(result)

        answer_id = row.get("answer_id", index)
        status = "OK" if result.get("judge_total_score") is not None else "ERROR"
        print(f"[{position}/{total_rows}] {status} answer_id={answer_id}", flush=True)

        if status == "ERROR":
            print(f"  {result.get('judge_comment', 'No error details')}", flush=True)

        # Keep batch requests gentle for API rate limits.
        time.sleep(0.5)

    judge_df = pd.DataFrame(rows, index=scored_df.index)

    for column in JUDGE_SCORE_COLUMNS:
        scored_df[column] = judge_df[column]

    return scored_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge scoring with DeepSeek API.")
    
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to input answers CSV.",
    )
    
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to output answers CSV.",
    )
    
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Path to JSON weights file (scoring_weights.json)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    weights_path = Path(args.weights)

    if not input_path.exists():
        print(f"Input CSV not found: {input_path}")
        return

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        print(f"Input CSV is empty: {input_path}")
        return

    if df.empty:
        print(f"Input CSV is empty: {input_path}")
        return

    weights = load_weights(weights_path)

    scored_df = score_dataframe_with_judge(df, weights=weights)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Judge scores saved to: {output_path}")
    print(f"Использованы веса из: {weights_path}")
    print(f"Rows scored: {len(scored_df)}")


if __name__ == "__main__":
    main()