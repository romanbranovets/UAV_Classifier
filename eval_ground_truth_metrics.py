#!/usr/bin/env python3
"""Classification metrics for NN and heuristic labels in ground_truth.json files.

Standalone utility — not part of the ML training pipeline.
Ground truth: ``Тип цели``; predictions: ``NN: класс`` and ``Эвристика: класс``.
Entries with unknown ground truth (``—``) are excluded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

CLASSES = ("Фон", "ДВС", "ЭД")
TARGET_CLASSES = ("ДВС", "ЭД")
GT_KEY = "Тип цели"
PREDICTORS = {
    "NN": "NN: класс",
    "Эвристика": "Эвристика: класс",
}

# Mapping from sklearn / modern ML terms to classic detection-theory wording.
SOVIET_TERMS = (
    ("Вероятность правильного решения", "P_пр.р.", "Accuracy", "Доля каналов, где класс определён верно"),
    ("Вероятность обнаружения", "P_обн", "Recall (полнота)", "Доля объектов данного класса, распознанных верно"),
    ("Вероятность пропуска цели", "P_пр", "1 − Recall", "Доля объектов класса, ошибочно отнесённых к другому классу"),
    ("Вероятность ложной тревоги на фоне", "P_лт⁰", "1 − P_обн(Фон)", "Доля фоновых каналов, ошибочно объявленных целью"),
    ("Вероятность ложной тревоги (по классу)", "P_лт", "FPR (one-vs-rest)", "Доля «чужих» объектов, ошибочно объявленных данным классом"),
    ("Достоверность тревоги", "—", "Precision", "Доля верных срабатываний среди всех объявлений данного класса"),
    ("Интегральная эффективность", "—", "F1", "Гармоническое среднее достоверности и полноты"),
    ("Матрица ошибок", "—", "Confusion matrix", "Таблица: истинный класс × объявленный класс"),
)


@dataclass(frozen=True)
class Sample:
    session: str
    ground_truth: str
    prediction: str


@dataclass
class PredictorMetrics:
    name: str
    n_evaluated: int
    n_skipped_no_pred: int
    accuracy: float
    macro_f1: float
    p_lt_on_background: float
    per_class: dict[str, dict[str, float]]
    confusion: np.ndarray
    classes: tuple[str, ...] = CLASSES


def iter_ground_truth_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/ground_truth.json"))


def load_samples(root: Path, pred_key: str) -> tuple[list[Sample], dict[str, int]]:
    samples: list[Sample] = []
    stats = {
        "entries_total": 0,
        "skipped_unknown_gt": 0,
        "skipped_missing_pred": 0,
        "skipped_invalid_pred": 0,
    }

    for path in iter_ground_truth_files(root):
        session = path.parent.name
        entries = json.loads(path.read_text(encoding="utf-8"))
        for entry in entries:
            stats["entries_total"] += 1
            gt = entry.get(GT_KEY, "")
            if gt not in CLASSES:
                stats["skipped_unknown_gt"] += 1
                continue

            pred = entry.get(pred_key)
            if pred is None:
                stats["skipped_missing_pred"] += 1
                continue
            if pred not in CLASSES:
                stats["skipped_invalid_pred"] += 1
                continue

            samples.append(Sample(session=session, ground_truth=gt, prediction=pred))

    return samples, stats


def false_alarm_rate(y_true: list[str], y_pred: list[str], target_class: str) -> float:
    negatives = [t for t in y_true if t != target_class]
    if not negatives:
        return float("nan")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != target_class and p == target_class)
    return fp / len(negatives)


def compute_metrics(
    name: str,
    samples: list[Sample],
    n_skipped_no_pred: int,
    *,
    classes: tuple[str, ...] = CLASSES,
) -> PredictorMetrics:
    y_true = [s.ground_truth for s in samples]
    y_pred = [s.prediction for s in samples]
    labels = list(classes)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    per_class: dict[str, dict[str, float]] = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            "support": float(support[i]),
            "p_obn": float(recall[i]),
            "p_pr": float(1.0 - recall[i]),
            "p_lt": float(false_alarm_rate(y_true, y_pred, cls)),
            "dostovernost": float(precision[i]),
            "f1": float(f1[i]),
        }

    p_lt_on_background = per_class["Фон"]["p_pr"] if "Фон" in per_class else float("nan")

    return PredictorMetrics(
        name=name,
        n_evaluated=len(samples),
        n_skipped_no_pred=n_skipped_no_pred,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        p_lt_on_background=p_lt_on_background,
        per_class=per_class,
        confusion=matrix,
        classes=classes,
    )


def filter_samples(samples: list[Sample], *, exclude_gt: set[str]) -> list[Sample]:
    return [s for s in samples if s.ground_truth not in exclude_gt]


def pct(value: float) -> str:
    if np.isnan(value):
        return "—"
    return f"{100.0 * value:.1f}%"


def format_confusion_matrix_md(matrix: np.ndarray, classes: tuple[str, ...]) -> str:
    header = "| Истина \\ Решение | " + " | ".join(classes) + " |"
    sep = "|" + "|".join(["---"] * (len(classes) + 1)) + "|"
    rows = [header, sep]
    for i, cls in enumerate(classes):
        cells = " | ".join(str(int(matrix[i, j])) for j in range(len(classes)))
        rows.append(f"| **{cls}** | {cells} |")
    return "\n".join(rows)


def build_markdown_report(
    root: Path,
    session_count: int,
    stats_by_predictor: dict[str, dict[str, int]],
    metrics: list[PredictorMetrics],
    heur_with_background: PredictorMetrics,
    targets_only_metrics: list[PredictorMetrics],
) -> str:
    nn_targets = next(m for m in targets_only_metrics if m.name.startswith("NN"))
    heur_targets = next(m for m in targets_only_metrics if m.name == "Эвристика")
    total_entries = next(iter(stats_by_predictor.values()))["entries_total"]
    skipped_unknown = next(iter(stats_by_predictor.values()))["skipped_unknown_gt"]

    lines = [
        "# Оценка классификаторов по размеченному датасету",
        "",
        f"**Дата:** {date.today().isoformat()}  ",
        f"**Датасет:** `{root.name}/` ({session_count} сессий)  ",
        f"**Размеченных каналов:** {total_entries - skipped_unknown} "
        f"(записи с типом цели «—» не учитываются: {skipped_unknown})",
        "",
        "## Краткие выводы",
        "",
    ]

    if len(metrics) == 2:
        nn, heur = metrics
        lines += [
            f"- **NN** (все классы): P_пр.р. = {pct(nn.accuracy)}; "
            f"**эвристика** (только ДВС/ЭД): P_пр.р. = {pct(heur.accuracy)}.",
            f"- На **одинаковой выборке целей** (без фона) NN лучше: "
            f"{pct(nn_targets.accuracy)} против {pct(heur_targets.accuracy)} у эвристики.",
            "- На фоне ложная тревога: "
            f"{pct(nn.p_lt_on_background)} (NN) и {pct(heur_with_background.p_lt_on_background)} (эвристика, справочно).",
            "- **Эвристика не выдаёт класс «Фон»** — фоновые каналы исключены из её основной оценки.",
            f"- У NN нет предсказания для {nn.n_skipped_no_pred} каналов — они исключены из оценки NN.",
            "",
        ]

    lines += [
        "## Сводная таблица",
        "",
        "| Показатель | " + " | ".join(m.name for m in metrics) + " |",
        "|---|" + "|".join(["---:"] * len(metrics)) + "|",
        f"| Оценено каналов | " + " | ".join(str(m.n_evaluated) for m in metrics) + " |",
        f"| P_пр.р. (вероятность правильного решения) | "
        + " | ".join(pct(m.accuracy) for m in metrics) + " |",
        f"| P_лт на фоне (ложная тревога при отсутствии цели) | "
        + " | ".join(
            pct(m.p_lt_on_background) if not np.isnan(m.p_lt_on_background) else "—"
            for m in metrics
        ) + " |",
        f"| Интегральная эффективность (macro F1) | "
        + " | ".join(pct(m.macro_f1) for m in metrics) + " |",
        "",
        "## Показатели по классам цели",
        "",
        "| Класс | Показатель | " + " | ".join(m.name for m in metrics) + " |",
        "|---|---|" + "|".join(["---:"] * len(metrics)) + "|",
    ]

    row_labels = (
        ("P_обн (вероятность обнаружения)", "p_obn"),
        ("P_пр (вероятность пропуска)", "p_pr"),
        ("P_лт (вероятность ложной тревоги)", "p_lt"),
        ("Достоверность тревоги", "dostovernost"),
    )
    all_classes_in_table: list[str] = []
    for m in metrics:
        for cls in m.classes:
            if cls not in all_classes_in_table:
                all_classes_in_table.append(cls)

    for cls in all_classes_in_table:
        for label, key in row_labels:
            values: list[str] = []
            for m in metrics:
                if cls in m.per_class:
                    values.append(pct(m.per_class[cls][key]))
                else:
                    values.append("—")
            lines.append(f"| {cls} | {label} | " + " | ".join(values) + " |")
        support_values: list[str] = []
        for m in metrics:
            if cls in m.per_class:
                support_values.append(str(int(m.per_class[cls]["support"])))
            else:
                support_values.append("—")
        lines.append(f"| {cls} | Число каналов (выборка) | " + " | ".join(support_values) + " |")

    lines += [
        "",
        "### Эвристика на целях (без класса «Фон»)",
        "",
        "Эвристика **никогда не выдаёт класс «Фон»**, поэтому оценка фона для неё бессмысленна. "
        "Ниже — сравнение на каналах, где истинный тип цели **ДВС** или **ЭД** "
        f"({heur_with_background.n_evaluated - heur_with_background.per_class['Фон']['support']:.0f} каналов, "
        f"исключено фоновых: {int(heur_with_background.per_class['Фон']['support'])}).",
        "",
        "| Показатель | " + " | ".join(m.name for m in targets_only_metrics) + " |",
        "|---|" + "|".join(["---:"] * len(targets_only_metrics)) + "|",
        f"| Оценено каналов | " + " | ".join(str(m.n_evaluated) for m in targets_only_metrics) + " |",
        f"| P_пр.р. (вероятность правильного решения) | "
        + " | ".join(pct(m.accuracy) for m in targets_only_metrics) + " |",
        f"| Интегральная эффективность (macro F1) | "
        + " | ".join(pct(m.macro_f1) for m in targets_only_metrics) + " |",
        "",
        "| Класс | Показатель | " + " | ".join(m.name for m in targets_only_metrics) + " |",
        "|---|---|" + "|".join(["---:"] * len(targets_only_metrics)) + "|",
    ]
    for cls in TARGET_CLASSES:
        for label, key in row_labels:
            values = " | ".join(pct(m.per_class[cls][key]) for m in targets_only_metrics)
            lines.append(f"| {cls} | {label} | {values} |")
        lines.append(
            f"| {cls} | Число каналов (выборка) | "
            + " | ".join(str(int(m.per_class[cls]["support"])) for m in targets_only_metrics)
            + " |"
        )

    for m in targets_only_metrics:
        lines += ["", f"**Матрица ошибок — {m.name}:**", "",
                  format_confusion_matrix_md(m.confusion, TARGET_CLASSES), ""]

    lines += ["", "## Матрицы ошибок (полная оценка)", ""]
    for m in metrics:
        lines += [f"### {m.name}", "", format_confusion_matrix_md(m.confusion, m.classes), ""]

    lines += [
        "## Соответствие терминов",
        "",
        "Термины из учебников по обнаружению сигналов (Тихонов, Ширяев и др.) "
        "и их аналоги в данном отчёте:",
        "",
        "| Термин (учебник) | Обозначение | Аналог в отчёте | Смысл |",
        "|---|---|---|---|",
    ]
    for soviet, symbol, modern, meaning in SOVIET_TERMS:
        lines.append(f"| {soviet} | {symbol} | {modern} | {meaning} |")

    lines += [
        "",
        "## Примечания к методике",
        "",
        "- **Эталон (истина):** поле `Тип цели` в `ground_truth.json`.",
        "- **Решение системы:** `NN: класс` или `Эвристика: класс`.",
        "- Классы: **Фон** (нет цели), **ДВС**, **ЭД**.",
        "- Для **эвристики** класс «Фон» не учитывается: алгоритм его не выдаёт.",
        "- P_лт для класса X: доля каналов, где истина ≠ X, но система объявила X "
        "(one-vs-rest, аналог условной вероятности ложной тревоги при гипотезе H₁ = «это X»).",
        "- Записи без разметки (`Тип цели` = «—») **не входят** в расчёт.",
        "",
    ]
    return "\n".join(lines)


def evaluate_predictor_console(name: str, samples: list[Sample], stats: dict[str, int]) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(f"{'=' * 72}")
    print(
        f"Записей в датасете: {stats['entries_total']}; "
        f"оценено: {len(samples)}; "
        f"исключено (тип «—»): {stats['skipped_unknown_gt']}; "
        f"исключено (нет предсказания): {stats['skipped_missing_pred']}"
    )
    if not samples:
        print("Нет данных для оценки.")
        return

    y_true = [s.ground_truth for s in samples]
    y_pred = [s.prediction for s in samples]
    print(f"\nP_пр.р. (accuracy): {accuracy_score(y_true, y_pred):.4f}")
    print("\nМатрица ошибок:")
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    header = "pred →".ljust(12) + "".join(f"{cls:>8}" for cls in CLASSES)
    print(header)
    for i, cls in enumerate(CLASSES):
        row = f"{cls:12}" + "".join(f"{matrix[i, j]:8d}" for j in range(len(CLASSES)))
        print(row)
    print("\n", classification_report(
        y_true, y_pred, labels=list(CLASSES), target_names=list(CLASSES), zero_division=0, digits=4,
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Метрики классификации NN и эвристики по ground_truth.json."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("Dataset"),
        help="Корневая папка с сессиями (по умолчанию: Dataset)",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("classification_report.md"),
        help="Путь для MD-отчёта (по умолчанию: classification_report.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    if not root.is_dir():
        raise SystemExit(f"Папка датасета не найдена: {root}")

    session_count = len(list(iter_ground_truth_files(root)))
    print(f"Датасет: {root}")
    print(f"Сессий: {session_count}")

    stats_by_predictor: dict[str, dict[str, int]] = {}
    all_metrics: list[PredictorMetrics] = []
    heur_with_background: PredictorMetrics | None = None
    nn_targets: list[Sample] = []
    heur_targets: list[Sample] = []

    for name, pred_key in PREDICTORS.items():
        samples, stats = load_samples(root, pred_key)
        stats_by_predictor[name] = stats
        evaluate_predictor_console(name, samples, stats)

        if name == "Эвристика":
            heur_with_background = compute_metrics(name, samples, stats["skipped_missing_pred"])
            heur_targets = filter_samples(samples, exclude_gt={"Фон"})
            all_metrics.append(
                compute_metrics(name, heur_targets, stats["skipped_missing_pred"], classes=TARGET_CLASSES)
            )
        else:
            nn_targets = filter_samples(samples, exclude_gt={"Фон"})
            all_metrics.append(compute_metrics(name, samples, stats["skipped_missing_pred"]))

    assert heur_with_background is not None
    nn_stats = stats_by_predictor["NN"]
    heur_stats = stats_by_predictor["Эвристика"]
    targets_only_metrics = [
        compute_metrics(
            "NN (только ДВС/ЭД)",
            nn_targets,
            nn_stats["skipped_missing_pred"],
            classes=TARGET_CLASSES,
        ),
        compute_metrics(
            "Эвристика",
            heur_targets,
            heur_stats["skipped_missing_pred"],
            classes=TARGET_CLASSES,
        ),
    ]

    report = build_markdown_report(
        root, session_count, stats_by_predictor, all_metrics, heur_with_background, targets_only_metrics,
    )
    args.report.write_text(report, encoding="utf-8")
    print(f"\nMD-отчёт сохранён: {args.report.resolve()}")


if __name__ == "__main__":
    main()
