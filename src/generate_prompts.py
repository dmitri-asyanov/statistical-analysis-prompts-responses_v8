import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.features.prompt_features import extract_prompt_features
from src.features.text_utils import normalize_text


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

SCHEMA_COLUMNS = [
    "prompt_id",
    "category",
    "category_label",
    "length_level",
    "has_context",
    "has_constraints",
    "role",
    "role_text",
    "context_text",
    "constraints_text",
    "template_text",
    "variable_values",
    "expected_response_type",
    "prompt_text",

    "char_count",
    "word_count",
    "sentence_count",
    "prompt_line_count",
    "prompt_paragraph_count",

    "has_question_form",
    "has_role",
    "has_format_instruction",
    "has_code_in_prompt",

    "constraint_count",
    "instruction_count",
    "directive_words_count",
    "technical_terms_count",

    "structure_score",
    "specificity_score",

    "created_at",
]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def pick_variable_values(variables: dict, template_text: str) -> dict:
    selected = {}

    for var_name, values in variables.items():
        placeholder = "{" + var_name + "}"
        if placeholder in template_text:
            selected[var_name] = random.choice(values)

    return selected


def render_template(template_text: str, variable_values: dict) -> str:
    rendered = template_text
    for key, value in variable_values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def build_constraints_block(config: dict, category: str) -> str:
    common_constraint = random.choice(config["constraints"]["common"])
    category_constraint = random.choice(config["constraints"][category])
    return "Ограничения:\n- " + common_constraint + "\n- " + category_constraint


def build_prompt(
    config: dict,
    category: str,
    length_level: str,
    has_context: int,
    has_constraints: int,
    role_key: str,
) -> tuple[str, str, str, str, dict, str]:
    role_text = config["roles"][role_key]
    context_text = random.choice(config["contexts"][category]) if has_context else ""

    template_item = random.choice(config["categories"][category]["templates"][length_level])
    template_text = template_item["text"]
    expected_response_type = template_item["expected_response_type"]

    variable_values = pick_variable_values(
        config["categories"][category]["variables"],
        template_text,
    )
    main_instruction = render_template(template_text, variable_values)

    parts = []

    if role_text:
        parts.append(role_text)

    if context_text:
        parts.append(f"Контекст: {context_text}")

    parts.append(main_instruction)

    constraints_text = ""
    if has_constraints:
        constraints_text = build_constraints_block(config, category)
        parts.append(constraints_text)

    parts.append(config["formats"][length_level])

    prompt_text = normalize_text("\n\n".join(parts))

    return (
        prompt_text,
        template_text,
        context_text,
        constraints_text,
        variable_values,
        expected_response_type,
    )


def validate_record(record: dict) -> None:
    missing = set(SCHEMA_COLUMNS) - set(record.keys())
    if missing:
        raise ValueError(f"Record missing fields: {missing}")


def generate_prompt_record(
    categories_config: dict,
    templates_config: dict,
    category: str,
    length_level: str,
    has_context: int,
    has_constraints: int,
    role_key: str,
) -> dict:
    (
        prompt_text,
        template_text,
        context_text,
        constraints_text,
        variable_values,
        expected_response_type,
    ) = build_prompt(
        config=templates_config,
        category=category,
        length_level=length_level,
        has_context=has_context,
        has_constraints=has_constraints,
        role_key=role_key,
    )

    role_text = templates_config["roles"][role_key]
    meta = {
        "has_context": has_context,
        "has_constraints": has_constraints,
        "role": role_key,
        "role_text": role_text,
    }
    feature_values = extract_prompt_features(prompt_text, meta)

    record = {
        "prompt_id": str(uuid.uuid4()),
        "category": category,
        "category_label": categories_config["categories"][category]["label"],
        "length_level": length_level,
        "has_context": has_context,
        "has_constraints": has_constraints,
        "role": role_key,
        "role_text": role_text,
        "context_text": context_text,
        "constraints_text": constraints_text,
        "template_text": template_text,
        "variable_values": json.dumps(variable_values, ensure_ascii=False),
        "expected_response_type": expected_response_type,
        "prompt_text": prompt_text,
        **feature_values,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    validate_record(record)
    return record


def generate_all_prompts(categories_config: dict, templates_config: dict) -> pd.DataFrame:
    rows = []

    categories = list(categories_config["categories"].keys())
    lengths = ["short", "medium", "long"]
    context_values = [0, 1]
    constraint_values = [0, 1]
    roles = list(templates_config["roles"].keys())

    for category in categories:
        for length_level in lengths:
            for has_context in context_values:
                for has_constraints in constraint_values:
                    for role_key in roles:
                        row = generate_prompt_record(
                            categories_config=categories_config,
                            templates_config=templates_config,
                            category=category,
                            length_level=length_level,
                            has_context=has_context,
                            has_constraints=has_constraints,
                            role_key=role_key,
                        )
                        rows.append(row)

    return pd.DataFrame(rows)


def save_raw_prompts(df: pd.DataFrame) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    missing = set(SCHEMA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[SCHEMA_COLUMNS]

    output_path = RAW_DATA_DIR / "prompts_raw.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path


def main() -> None:
    random.seed(42)

    categories_config = load_json(CONFIGS_DIR / "categories.json")
    templates_config = load_json(CONFIGS_DIR / "templates.json")

    df = generate_all_prompts(categories_config, templates_config)
    output_path = save_raw_prompts(df)

    print(f"Generated {len(df)} prompts")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()