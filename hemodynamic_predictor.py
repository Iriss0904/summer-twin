from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any


DEFAULT_DATA_PATH = Path(__file__).with_name("dataset_40k_cases.csv")
DEFAULT_SUMMARY_INFO_PATH = Path(__file__).with_name("hemodynamic_summary.csv")

DEFAULT_TOP_K = 3
DEFAULT_TOLERANCE = 1.0
MIN_HARD_MATCHES = 10
K_NEIGHBORS = 500
INPUT_BANDWIDTH = 0.35
OUTPUT_BANDWIDTH = 1.2
N_CLUSTERS = 3
PCA_COMPONENTS = 6
RANDOM_STATE = 42


def _to_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: list[float], fallback: float = 1.0) -> float:
    if not values:
        return fallback
    mu = _mean(values)
    var = sum((x - mu) ** 2 for x in values) / len(values)
    std = math.sqrt(var)
    return std if std > 1e-12 else fallback


def _normalize_scores(items: list[dict[str, Any]]) -> None:
    total = sum(max(float(x.get("raw_score", 0.0)), 0.0) for x in items)
    for item in items:
        raw = max(float(item.get("raw_score", 0.0)), 0.0)
        item["confidence_score"] = raw / total if total > 0 else 0.0


def _squared_distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm <= 1e-12:
        return [0.0 for _ in vector]
    return [x / norm for x in vector]


def _standardize_matrix(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> tuple[list[list[float]], list[float], list[float]]:
    matrix = [[float(row[col]) for col in columns] for row in rows]
    means = [_mean([row[j] for row in matrix]) for j in range(len(columns))]
    stds = [_safe_std([row[j] for row in matrix]) for j in range(len(columns))]
    z = [
        [(row[j] - means[j]) / stds[j] for j in range(len(columns))]
        for row in matrix
    ]
    return z, means, stds


def _pca_project(
    matrix: list[list[float]],
    n_components: int,
) -> tuple[list[list[float]], str]:
    if not matrix or not matrix[0]:
        return matrix, "pca_skipped_empty"

    n = len(matrix)
    d = len(matrix[0])
    n_components = min(n_components, d, max(n - 1, 1))
    if n_components <= 0 or d <= n_components:
        return matrix, "pca_skipped_low_dimensional"

    denom = max(n - 1, 1)
    cov = [
        [
            sum(row[i] * row[j] for row in matrix) / denom
            for j in range(d)
        ]
        for i in range(d)
    ]

    total_var = sum(cov[i][i] for i in range(d))
    work = [row[:] for row in cov]
    components: list[list[float]] = []
    explained = 0.0
    rng = random.Random(RANDOM_STATE)

    for _ in range(n_components):
        vector = _unit([rng.random() - 0.5 for _ in range(d)])
        if not any(vector):
            break

        for _ in range(80):
            next_vector = _unit(_matvec(work, vector))
            if not any(next_vector):
                break
            if math.sqrt(_squared_distance(vector, next_vector)) < 1e-8:
                vector = next_vector
                break
            vector = next_vector

        eigenvalue = _dot(vector, _matvec(work, vector))
        if eigenvalue <= 1e-9:
            break

        components.append(vector)
        explained += eigenvalue

        for i in range(d):
            for j in range(d):
                work[i][j] -= eigenvalue * vector[i] * vector[j]

    if not components:
        return matrix, "pca_skipped_no_stable_components"

    projected = [[_dot(row, component) for component in components] for row in matrix]
    ratio = explained / total_var if total_var > 0 else 0.0
    return projected, f"pca_{len(components)}d_explained_var_{ratio:.3f}"


def _weighted_kmeans(
    features: list[list[float]],
    weights: list[float],
    n_clusters: int,
    max_iter: int = 60,
) -> tuple[list[int], list[list[float]]]:
    if not features:
        return [], []

    n_clusters = min(n_clusters, len(features))
    centers: list[list[float]] = []
    first = max(range(len(features)), key=lambda i: weights[i])
    centers.append(features[first][:])

    while len(centers) < n_clusters:
        next_idx = max(
            range(len(features)),
            key=lambda i: min(_squared_distance(features[i], c) for c in centers) * weights[i],
        )
        centers.append(features[next_idx][:])

    labels = [0 for _ in features]

    for _ in range(max_iter):
        changed = False
        for i, feature in enumerate(features):
            label = min(range(n_clusters), key=lambda c: _squared_distance(feature, centers[c]))
            if label != labels[i]:
                labels[i] = label
                changed = True

        new_centers: list[list[float]] = []
        for cluster_id in range(n_clusters):
            idx = [i for i, label in enumerate(labels) if label == cluster_id]
            if not idx:
                replacement = max(
                    range(len(features)),
                    key=lambda i: min(_squared_distance(features[i], c) for c in centers),
                )
                new_centers.append(features[replacement][:])
                continue

            total_weight = sum(weights[i] for i in idx)
            if total_weight <= 0:
                total_weight = float(len(idx))
                cluster_weights = [1.0 for _ in idx]
            else:
                cluster_weights = [weights[i] for i in idx]

            center = [
                sum(features[i][dim] * cluster_weights[pos] for pos, i in enumerate(idx))
                / total_weight
                for dim in range(len(features[0]))
            ]
            new_centers.append(center)

        shift = sum(_squared_distance(a, b) for a, b in zip(centers, new_centers))
        centers = new_centers
        if not changed or shift < 1e-8:
            break

    return labels, centers


def _expand_metric_label(label: str, known_columns: set[str]) -> list[str]:
    if label in known_columns:
        return [label]
    if "/" not in label:
        return []

    parts = [part.strip() for part in label.split("/") if part.strip()]
    expanded = [part for part in parts if part in known_columns]

    first = parts[0] if parts else ""
    prefix = first.rsplit("_", 1)[0] + "_" if "_" in first else ""
    if prefix:
        expanded.extend(prefix + part for part in parts[1:] if "_" not in part)

    return [col for col in dict.fromkeys(expanded) if col in known_columns]


class HemodynamicPredictor:
    def __init__(
        self,
        data_path: Path | str = DEFAULT_DATA_PATH,
        summary_info_path: Path | str = DEFAULT_SUMMARY_INFO_PATH,
    ) -> None:
        self.data_path = Path(data_path)
        self.summary_info_path = Path(summary_info_path)
        self.rows: list[dict[str, Any]] = []
        self.summary_columns: list[str] = []
        self.summary_info: dict[str, dict[str, str]] = {}
        self.column_stats: dict[str, dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        with self.data_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {self.data_path}")

            fieldnames = reader.fieldnames
            self.summary_columns = [
                col for col in fieldnames if col != "case_id" and not col.startswith("param_")
            ]
            if not self.summary_columns:
                raise ValueError("No hemodynamic summary columns found in dataset.")

            for raw in reader:
                row: dict[str, Any] = {"case_id": raw.get("case_id", "")}
                for col in self.summary_columns:
                    row[col] = _to_float(raw.get(col))
                self.rows.append(row)

        self.summary_info = self._load_summary_info()
        for col in self.summary_columns:
            values = [row[col] for row in self.rows if not math.isnan(row[col])]
            self.column_stats[col] = {
                "min": min(values) if values else math.nan,
                "max": max(values) if values else math.nan,
                "mean": _mean(values) if values else math.nan,
                "std": _safe_std(values) if values else 1.0,
            }

    def _load_summary_info(self) -> dict[str, dict[str, str]]:
        info = {
            col: {"name": col, "zh": "", "en": "", "unit": "", "clinical": ""}
            for col in self.summary_columns
        }
        if not self.summary_info_path.exists():
            return info

        known = set(self.summary_columns)
        with self.summary_info_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = (row.get("指标名") or "").strip()
                if not label:
                    continue
                for col in _expand_metric_label(label, known):
                    info[col] = {
                        "name": col,
                        "zh": (row.get("中文名") or "").strip(),
                        "en": (row.get("英文名") or "").strip(),
                        "unit": (row.get("单位") or "").strip(),
                        "clinical": (row.get("临床意义") or "").strip(),
                    }
        return info

    def metadata(self) -> dict[str, Any]:
        return {
            "row_count": len(self.rows),
            "summary_columns": [
                {
                    **self.summary_info.get(col, {"name": col}),
                    "name": col,
                    "min": self.column_stats[col]["min"],
                    "max": self.column_stats[col]["max"],
                }
                for col in self.summary_columns
            ],
            "defaults": {
                "inputs": {"SBP": 100.0, "MAP": 90.0},
                "outputs": ["sPAP", "dPAP"],
                "tolerance": DEFAULT_TOLERANCE,
                "top_k": DEFAULT_TOP_K,
            },
        }

    def predict(
        self,
        inputs: dict[str, float],
        outputs: list[str],
        tolerance: float = DEFAULT_TOLERANCE,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        clean_inputs = self._validate_inputs(inputs)
        clean_outputs = self._validate_outputs(outputs, clean_inputs)
        tolerance = float(tolerance)
        top_k = max(1, int(top_k))

        needed_cols = list(clean_inputs) + clean_outputs
        valid_rows = [
            row
            for row in self.rows
            if all(not math.isnan(row[col]) for col in needed_cols)
        ]
        if not valid_rows:
            raise ValueError("No rows have complete values for the selected columns.")

        hard_matches = [
            row
            for row in valid_rows
            if all(abs(row[col] - target) <= tolerance for col, target in clean_inputs.items())
        ]

        if len(hard_matches) >= MIN_HARD_MATCHES:
            candidate_source = "hard_matches"
            candidates = self._with_input_distance(hard_matches, clean_inputs)
            candidates.sort(key=lambda row: row["input_distance"])
            candidates = candidates[:K_NEIGHBORS]
            for row in candidates:
                row["input_weight"] = 1.0
        else:
            candidate_source = "weighted_neighbors"
            candidates = self._weighted_neighbors(valid_rows, clean_inputs, K_NEIGHBORS)

        if not candidates:
            raise ValueError("No candidate rows found. Try a wider tolerance or fewer inputs.")

        n_outputs = len(clean_outputs)
        if n_outputs <= 4:
            top, mode = self._conditional_kde(candidates, clean_outputs, top_k)
        elif n_outputs <= 10:
            top, mode = self._weighted_cluster_medoids(
                candidates,
                clean_outputs,
                top_k,
                use_pca=False,
            )
        else:
            top, mode = self._weighted_cluster_medoids(
                candidates,
                clean_outputs,
                top_k,
                use_pca=True,
            )

        return {
            "inputs": clean_inputs,
            "outputs": clean_outputs,
            "tolerance": tolerance,
            "mode": mode,
            "candidate_source": candidate_source,
            "row_count": len(valid_rows),
            "hard_match_count": len(hard_matches),
            "candidate_count": len(candidates),
            "top": [self._serialize_case(row, clean_inputs, clean_outputs) for row in top],
        }

    def _validate_inputs(self, inputs: dict[str, float]) -> dict[str, float]:
        clean: dict[str, float] = {}
        known = set(self.summary_columns)
        for col, value in inputs.items():
            if col not in known:
                raise ValueError(f"Unknown input summary: {col}")
            parsed = _to_float(value)
            if not math.isnan(parsed):
                clean[col] = parsed
        if not clean:
            raise ValueError("At least one numeric input summary is required.")
        return clean

    def _validate_outputs(
        self,
        outputs: list[str],
        inputs: dict[str, float],
    ) -> list[str]:
        known = set(self.summary_columns)
        clean = []
        for col in outputs:
            if col not in known:
                raise ValueError(f"Unknown output summary: {col}")
            if col in inputs:
                raise ValueError(f"Output summary already used as input: {col}")
            if col not in clean:
                clean.append(col)
        if not clean:
            raise ValueError("At least one output summary must be selected.")
        return clean

    def _input_distance(self, row: dict[str, Any], targets: dict[str, float]) -> float:
        total = 0.0
        for col, target in targets.items():
            std = self.column_stats[col]["std"] or 1.0
            total += ((row[col] - target) / std) ** 2
        return math.sqrt(total)

    def _with_input_distance(
        self,
        rows: list[dict[str, Any]],
        targets: dict[str, float],
    ) -> list[dict[str, Any]]:
        weighted = []
        for row in rows:
            item = dict(row)
            item["input_distance"] = self._input_distance(row, targets)
            item["input_weight"] = math.exp(
                -0.5 * (item["input_distance"] / INPUT_BANDWIDTH) ** 2
            )
            weighted.append(item)
        return weighted

    def _weighted_neighbors(
        self,
        rows: list[dict[str, Any]],
        targets: dict[str, float],
        k: int,
    ) -> list[dict[str, Any]]:
        weighted = self._with_input_distance(rows, targets)
        weighted.sort(key=lambda row: row["input_distance"])
        return weighted[: min(k, len(weighted))]

    def _conditional_kde(
        self,
        candidates: list[dict[str, Any]],
        outputs: list[str],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], str]:
        z, _, _ = _standardize_matrix(candidates, outputs)
        weights = [float(row.get("input_weight", 1.0)) for row in candidates]
        scored = []

        for i, row in enumerate(candidates):
            density = 0.0
            for j, other_z in enumerate(z):
                dist2 = _squared_distance(z[i], other_z)
                density += weights[j] * math.exp(-0.5 * dist2 / (OUTPUT_BANDWIDTH**2))

            item = dict(row)
            item["output_density"] = density
            item["raw_score"] = weights[i] * density
            scored.append(item)

        _normalize_scores(scored)
        scored.sort(key=lambda row: row["confidence_score"], reverse=True)
        return scored[:top_k], "conditional_kde_real_samples"

    def _weighted_cluster_medoids(
        self,
        candidates: list[dict[str, Any]],
        outputs: list[str],
        top_k: int,
        use_pca: bool,
    ) -> tuple[list[dict[str, Any]], str]:
        z, _, _ = _standardize_matrix(candidates, outputs)
        transform = "standardized_output_space"
        if use_pca:
            z, transform = _pca_project(z, PCA_COMPONENTS)

        weights = [float(row.get("input_weight", 1.0)) for row in candidates]
        labels, centers = _weighted_kmeans(z, weights, N_CLUSTERS)
        total_weight = sum(weights)
        possibilities: list[dict[str, Any]] = []

        for cluster_id, center in enumerate(centers):
            idx = [i for i, label in enumerate(labels) if label == cluster_id]
            if not idx:
                continue

            cluster_weight = sum(weights[i] for i in idx)
            medoid_idx = min(
                idx,
                key=lambda i: math.sqrt(_squared_distance(z[i], center))
                / max(weights[i], 1e-12),
            )

            item = dict(candidates[medoid_idx])
            item["cluster_id"] = cluster_id
            item["cluster_size"] = len(idx)
            item["cluster_weight"] = cluster_weight
            item["transform"] = transform
            item["raw_score"] = cluster_weight / total_weight if total_weight > 0 else 0.0
            possibilities.append(item)

        _normalize_scores(possibilities)
        possibilities.sort(key=lambda row: row["confidence_score"], reverse=True)
        mode = "pca_weighted_cluster_medoids" if use_pca else "weighted_cluster_medoids"
        return possibilities[:top_k], mode

    def _serialize_case(
        self,
        row: dict[str, Any],
        inputs: dict[str, float],
        outputs: list[str],
    ) -> dict[str, Any]:
        item = {
            "case_id": row["case_id"],
            "confidence_score": row.get("confidence_score", 0.0),
            "input_distance": row.get("input_distance", 0.0),
            "input_weight": row.get("input_weight", 1.0),
            "input_values": {col: row[col] for col in inputs},
            "outputs": {col: row[col] for col in outputs},
        }
        for key in [
            "output_density",
            "cluster_id",
            "cluster_size",
            "cluster_weight",
            "transform",
        ]:
            if key in row:
                item[key] = row[key]
        return item
