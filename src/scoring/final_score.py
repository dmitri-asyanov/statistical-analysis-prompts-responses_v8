"""
Final scoring script.

Reads the weights from weights.json and calculates the `final_score`
based on task_score, formal_total_score, and judge_total_score.
"""

import argparse
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "processed" / "answers.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "answers.csv"
DEFAULT_WEIGHTS_PATH =  BASE_DIR / "configs" / "scoring_weights.json" 


def load_weights(weights_path: Path) -> dict:
    if not weights_path.exists():
        raise FileNotFoundError(f"Файл весов не найден: {weights_path}")
    with open(weights_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_final_score(row: pd.Series, weights: dict) -> float:
    # Безопасное извлечение метрик с подменой NaN на 0.0
    task = float(row.get("task_score", 0.0)) if pd.notna(row.get("task_score")) else 0.0
    formal = float(row.get("formal_total_score", 0.0)) if pd.notna(row.get("formal_total_score")) else 0.0
    
    # Judge score идет от 0 до 10, нужно нормализовать до 0..1
    judge_raw = float(row.get("judge_total_score", 0.0)) if pd.notna(row.get("judge_total_score")) else 0.0
    judge = judge_raw / 10.0

    # Получаем веса из конфига (с фоллбеком на дефолтные значения)
    final_weights = weights.get("final_score", {})
    w_task = final_weights.get("task_score_weight", 0.35)
    w_formal = final_weights.get("formal_score_weight", 0.15)
    w_judge = final_weights.get("judge_score_weight", 0.50)

    # Взвешенная сумма
    final_score = (task * w_task) + (formal * w_formal) + (judge * w_judge)
    
    # Ограничиваем в рамках 0.0 - 1.0 на всякий случай
    return round(max(0.0, min(1.0, final_score)), 4)


def apply_final_score(input_path: str | Path, output_path: str | Path, weights_path: str | Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    weights_path = Path(weights_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")

    print("Загрузка данных и весов...")
    weights = load_weights(weights_path)
    df = pd.read_csv(input_path)

    # Проверка наличия нужных колонок
    required_cols = {"task_score", "formal_total_score", "judge_total_score"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ВНИМАНИЕ: Отсутствуют столбцы {missing}. Для этих метрик будет использовано значение 0.0.")

    print("Расчет final_score...")
    df["final_score"] = df.apply(lambda row: calculate_final_score(row, weights), axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Final score успешно рассчитан и сохранен: {output_path}")
    print("\nРаспределение final_score:")
    print(df["final_score"].describe().round(4))
    
    if "model_name" in df.columns:
        print("\nСредний final_score по моделям:")
        print(df.groupby("model_name")["final_score"].mean().round(4).sort_values(ascending=False).to_string())


def main():
    parser = argparse.ArgumentParser(description="Вычисление итоговой оценки (final_score) на основе агрегации других оценок.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Путь к answers.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Куда сохранить результат")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH), help="Путь к weights.json")

    args = parser.parse_args()

    apply_final_score(args.input, args.output, args.weights)


if __name__ == "__main__":
    main()