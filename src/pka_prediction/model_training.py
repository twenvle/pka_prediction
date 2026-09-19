#!/usr/bin/env python3
"""Nested CVによる回帰モデル学習をNotebookとCLIから共用する。"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# この版で使用する量子化学・分子記述子。
FEATURE_COLUMNS = (
    "polar",
    "h_nbo_charge",
    "o_nbo_charge",
    "total_nbo_charge",
    "dipole_moment_debye",
    "homo_ev",
    "lumo_ev",
    "gap_ev",
    "MW",
    "logP",
    "HBA",
    "HBD",
    "rotatable_bonds",
    "aromatic_rings",
    "FractionCSP3",
    "acid_group_count",
)

MODEL_NAMES = ("xgboost", "svr", "random_forest", "ridge")


@dataclass(frozen=True)
class TrainingConfig:
    """Nested CVを含む学習条件。

    ``model_parameters`` は探索しない推定器の固定設定、
    ``hyperparameter_grid`` は内側CVで探索する候補を表す。
    探索候補のキーは ``alpha`` と ``model__alpha`` のどちらでも指定できる。
    """

    model_name: str
    input_path: Path | str
    target_column: str
    output_dir: Path | str | None = None
    n_splits: int = 5
    inner_splits: int = 5
    random_state: int = 42
    encoding: str = "utf-8-sig"
    model_parameters: Mapping[str, Any] = field(default_factory=dict)
    hyperparameter_grid: Mapping[str, Sequence[Any]] | None = None

    def __post_init__(self) -> None:
        if self.model_name not in MODEL_NAMES:
            raise ValueError(
                f"未対応のモデルです: {self.model_name} "
                f"（選択肢: {', '.join(MODEL_NAMES)}）"
            )
        if not str(self.target_column).strip():
            raise ValueError("target_column を指定してください")
        if self.n_splits < 2:
            raise ValueError("n_splits は2以上にしてください")
        if self.inner_splits < 2:
            raise ValueError("inner_splits は2以上にしてください")
        if not isinstance(self.model_parameters, Mapping):
            raise TypeError("model_parameters は辞書形式で指定してください")
        if self.hyperparameter_grid is not None and not isinstance(
            self.hyperparameter_grid, Mapping
        ):
            raise TypeError("hyperparameter_grid は辞書形式で指定してください")

        normalized_grid: dict[str, list[Any]] | None = None
        if self.hyperparameter_grid is not None:
            normalized_grid = {}
            for name, candidates in self.hyperparameter_grid.items():
                if isinstance(candidates, (str, bytes)) or not isinstance(
                    candidates, Sequence
                ):
                    raise TypeError(
                        f"hyperparameter_grid.{name} は候補値のリストにしてください"
                    )
                if not candidates:
                    raise ValueError(f"hyperparameter_grid.{name} に候補値がありません")
                normalized_grid[str(name)] = list(candidates)
            if not normalized_grid:
                raise ValueError("hyperparameter_grid を空にすることはできません")

        object.__setattr__(self, "input_path", Path(self.input_path))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "model_parameters", dict(self.model_parameters))
        object.__setattr__(self, "hyperparameter_grid", normalized_grid)

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        return Path("model_results") / self.model_name

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: Path | str | None = None,
    ) -> "TrainingConfig":
        """辞書から設定を生成し、相対パスを設定ファイル基準で解決する。"""
        base = Path.cwd() if base_dir is None else Path(base_dir)
        project_root = _resolve_path(values.get("project_root", "."), base)

        model_name = values.get("model_name", values.get("model"))
        if not model_name:
            raise ValueError("model_name を指定してください")
        model_name = str(model_name)

        input_value = values.get("input_path", values.get("input"))
        if not input_value:
            raise ValueError("input_path を指定してください")
        target_column = values.get("target_column")
        if not target_column:
            raise ValueError("target_column を指定してください")

        output_value = values.get("output_dir")
        if output_value:
            output_dir = _resolve_path(output_value, project_root)
        else:
            output_root = values.get("output_root")
            output_dir = (
                _resolve_path(output_root, project_root) / model_name
                if output_root
                else project_root / "model_results" / model_name
            )

        model_parameters = _select_model_mapping(
            values.get("model_parameters", {}), model_name, "model_parameters"
        )
        raw_grid = values.get("hyperparameter_grid", values.get("hyperparameter_grids"))
        hyperparameter_grid = (
            None
            if raw_grid is None
            else _select_model_mapping(
                raw_grid, model_name, "hyperparameter_grid", allow_missing_model=True
            )
        )

        return cls(
            model_name=model_name,
            input_path=_resolve_path(input_value, project_root),
            target_column=str(target_column),
            output_dir=output_dir,
            n_splits=int(values.get("n_splits", 5)),
            inner_splits=int(values.get("inner_splits", 5)),
            random_state=int(values.get("random_state", 42)),
            encoding=str(values.get("encoding", "utf-8-sig")),
            model_parameters=dict(model_parameters),
            hyperparameter_grid=(
                None if hyperparameter_grid is None else dict(hyperparameter_grid)
            ),
        )


@dataclass
class TrainingResult:
    """Notebookから後続処理に利用できるNested CV学習結果。"""

    config: TrainingConfig
    pipeline: Any
    metrics: dict[str, Any]
    cv_predictions: Any
    model_path: Path | None = None
    metrics_path: Path | None = None
    predictions_path: Path | None = None


def _select_model_mapping(
    value: Any,
    model_name: str,
    setting_name: str,
    *,
    allow_missing_model: bool = False,
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{setting_name} は辞書形式で指定してください")
    if any(name in value for name in MODEL_NAMES):
        selected = value.get(model_name)
        if selected is None and allow_missing_model:
            return None
        if selected is None:
            selected = {}
    else:
        selected = value
    if not isinstance(selected, Mapping):
        raise TypeError(f"{setting_name}.{model_name} は辞書形式で指定してください")
    return selected


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _read_config_mapping(config_path: Path | str) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML設定の読み込みにはPyYAMLが必要です。次のコマンドで"
            "導入してください: pip install PyYAML"
        ) from exc

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"設定ファイルのYAMLが不正です: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("設定ファイルの最上位はYAMLのマッピング形式にしてください")
    return raw, path


def load_training_config(
    config_path: Path | str,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> TrainingConfig:
    """YAML設定を読み込み、任意の値で上書きして学習条件を返す。"""
    values, path = _read_config_mapping(config_path)
    if overrides:
        values.update(overrides)
    return TrainingConfig.from_mapping(values, base_dir=path.parent)


def _import_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import pandas as pd
        import sklearn
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import GridSearchCV, KFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
    except ImportError as exc:
        raise RuntimeError(
            "学習に必要なパッケージがありません。次のコマンドで導入してください:\n"
            "pip install pandas scikit-learn joblib xgboost"
        ) from exc

    return {
        "joblib": joblib,
        "pd": pd,
        "sklearn": sklearn,
        "RandomForestRegressor": RandomForestRegressor,
        "SimpleImputer": SimpleImputer,
        "Ridge": Ridge,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "GridSearchCV": GridSearchCV,
        "KFold": KFold,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "SVR": SVR,
    }


def _build_pipeline(
    model_name: str,
    deps: Mapping[str, Any],
    random_state: int,
    model_parameters: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, str]:
    imputer = deps["SimpleImputer"](strategy="median")
    overrides = dict(model_parameters or {})

    if model_name == "xgboost":
        try:
            import xgboost
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "XGBoostがありません。次のコマンドで導入してください: "
                "pip install xgboost"
            ) from exc
        parameters = {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": 1,
        }
        parameters.update(overrides)
        estimator = XGBRegressor(**parameters)
        steps = [("imputer", imputer), ("model", estimator)]
        package_version = xgboost.__version__
    elif model_name == "svr":
        parameters = {
            "kernel": "rbf",
            "C": 10.0,
            "epsilon": 0.1,
            "gamma": "scale",
        }
        parameters.update(overrides)
        estimator = deps["SVR"](**parameters)
        steps = [
            ("imputer", imputer),
            ("scaler", deps["StandardScaler"]()),
            ("model", estimator),
        ]
        package_version = deps["sklearn"].__version__
    elif model_name == "random_forest":
        parameters = {
            "n_estimators": 500,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "random_state": random_state,
            "n_jobs": 1,
        }
        parameters.update(overrides)
        estimator = deps["RandomForestRegressor"](**parameters)
        steps = [("imputer", imputer), ("model", estimator)]
        package_version = deps["sklearn"].__version__
    elif model_name == "ridge":
        parameters = {"alpha": 1.0}
        parameters.update(overrides)
        estimator = deps["Ridge"](**parameters)
        steps = [
            ("imputer", imputer),
            ("scaler", deps["StandardScaler"]()),
            ("model", estimator),
        ]
        package_version = deps["sklearn"].__version__
    else:
        raise ValueError(f"未対応のモデルです: {model_name}")

    return deps["Pipeline"](steps), estimator, package_version


def _default_hyperparameter_grid(model_name: str) -> dict[str, list[Any]]:
    if model_name == "xgboost":
        return {
            "model__n_estimators": [200, 400, 600, 800],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 5, 6],
            "model__min_child_weight": [1, 3, 5, 10],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "model__reg_lambda": [0.1, 1.0, 5.0, 10.0],
        }
    if model_name == "svr":
        return {
            "model__C": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            "model__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
        }
    if model_name == "random_forest":
        return {
            "model__n_estimators": [200, 500, 800],
            "model__max_depth": [None, 5, 10, 20],
            "model__max_features": ["sqrt", 0.5, 1.0],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__bootstrap": [True, False],
        }
    if model_name == "ridge":
        return {
            "model__alpha": [
                0.0001,
                0.0003,
                0.001,
                0.003,
                0.01,
                0.03,
                0.1,
                0.3,
                1.0,
                3.0,
                10.0,
                30.0,
                100.0,
                300.0,
                1000.0,
                3000.0,
                10000.0,
            ]
        }
    raise ValueError(f"未対応のモデルです: {model_name}")


def _normalize_hyperparameter_grid(
    grid: Mapping[str, Sequence[Any]],
) -> dict[str, list[Any]]:
    return {
        (str(name) if "__" in str(name) else f"model__{name}"): list(candidates)
        for name, candidates in grid.items()
    }


def _build_hyperparameter_search(
    model_name: str,
    pipeline: Any,
    deps: Mapping[str, Any],
    inner_splits: int,
    random_state: int,
    hyperparameter_grid: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    """外側学習データ内で使用する内側K-Fold探索器を作る。"""
    inner_cv = deps["KFold"](
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_state,
    )
    parameters = (
        _default_hyperparameter_grid(model_name)
        if hyperparameter_grid is None
        else _normalize_hyperparameter_grid(hyperparameter_grid)
    )
    search = deps["GridSearchCV"](
        estimator=pipeline,
        param_grid=parameters,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    return search, "GridSearchCV", parameters


def _resolve_column(columns: Sequence[str], requested: str) -> str:
    matches = [
        column for column in columns if column.casefold() == requested.casefold()
    ]
    if not matches:
        raise ValueError(f"列 '{requested}' が入力CSVに見つかりません")
    if len(matches) > 1:
        raise ValueError(f"大文字・小文字だけが異なる重複列があります: {requested}")
    return matches[0]


def _prepare_data(
    input_path: Path,
    target_column: str,
    encoding: str,
    deps: Mapping[str, Any],
) -> dict[str, Any]:
    pd = deps["pd"]
    frame = pd.read_csv(input_path, encoding=encoding)
    if frame.empty:
        raise ValueError("入力CSVにデータ行がありません")

    actual_target = _resolve_column(list(frame.columns), target_column)
    actual_features: dict[str, str] = {}
    missing_features: list[str] = []
    for feature in FEATURE_COLUMNS:
        try:
            actual_features[feature] = _resolve_column(list(frame.columns), feature)
        except ValueError:
            missing_features.append(feature)
    if missing_features:
        raise ValueError(
            "次の特徴量列が入力CSVにありません: " + ", ".join(missing_features)
        )
    if actual_target.casefold() in {
        column.casefold() for column in actual_features.values()
    }:
        raise ValueError("目的変数に特徴量列そのものを指定することはできません")

    numeric_target = pd.to_numeric(frame[actual_target], errors="coerce")
    valid_mask = numeric_target.notna()
    invalid_smiles_rows = 0
    try:
        error_column = _resolve_column(list(frame.columns), "error")
    except ValueError:
        error_column = None
    if error_column is not None:
        valid_smiles = frame[error_column].fillna("").astype(str).str.strip().eq("")
        invalid_smiles_rows = int((~valid_smiles).sum())
        valid_mask &= valid_smiles

    valid_indices = frame.index[valid_mask]
    if len(valid_indices) < 4:
        raise ValueError(
            "有効な学習データが少なすぎます。目的変数が数値である有効行を"
            "4行以上用意してください"
        )

    feature_frame = pd.DataFrame(index=valid_indices)
    for standard_name, actual_name in actual_features.items():
        feature_frame[standard_name] = pd.to_numeric(
            frame.loc[valid_indices, actual_name], errors="coerce"
        )

    all_missing_features = [
        column for column in FEATURE_COLUMNS if feature_frame[column].isna().all()
    ]
    for column in all_missing_features:
        feature_frame[column] = 0.0

    target = numeric_target.loc[valid_indices].astype(float)
    return {
        "frame": frame,
        "X": feature_frame,
        "y": target,
        "target_column": actual_target,
        "rows_read": len(frame),
        "rows_used": len(valid_indices),
        "rows_excluded": len(frame) - len(valid_indices),
        "rows_with_invalid_smiles": invalid_smiles_rows,
        "all_missing_features_filled_with_zero": all_missing_features,
    }


def _metric_values(
    y_true: Any, y_pred: Any, deps: Mapping[str, Any]
) -> dict[str, float]:
    mse = deps["mean_squared_error"](y_true, y_pred)
    return {
        "r2": float(deps["r2_score"](y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
        "mae": float(deps["mean_absolute_error"](y_true, y_pred)),
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _search_candidate_results(search: Any) -> list[dict[str, Any]]:
    cv_results = search.cv_results_
    candidates = [
        {
            "rank": int(rank),
            "mean_cv_mae": float(-mean_score),
            "std_cv_mae": float(std_score),
            "parameters": _json_value(parameters),
        }
        for rank, mean_score, std_score, parameters in zip(
            cv_results["rank_test_score"],
            cv_results["mean_test_score"],
            cv_results["std_test_score"],
            cv_results["params"],
        )
    ]
    candidates.sort(key=lambda item: item["rank"])
    return candidates


def _ridge_feature_contributions(pipeline: Any) -> dict[str, Any]:
    ridge_model = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]
    standardized_coefficients = [float(value) for value in ridge_model.coef_]
    original_scale_coefficients = [
        coefficient / float(scale)
        for coefficient, scale in zip(standardized_coefficients, scaler.scale_)
    ]
    standardized_intercept = float(ridge_model.intercept_)
    original_scale_intercept = standardized_intercept - sum(
        coefficient * float(mean) / float(scale)
        for coefficient, mean, scale in zip(
            standardized_coefficients, scaler.mean_, scaler.scale_
        )
    )
    details = [
        {
            "feature": feature,
            "standardized_coefficient": coefficient,
            "absolute_standardized_coefficient": abs(coefficient),
            "original_scale_coefficient": original_coefficient,
            "direction": (
                "positive"
                if coefficient > 0
                else "negative" if coefficient < 0 else "zero"
            ),
        }
        for feature, coefficient, original_coefficient in zip(
            FEATURE_COLUMNS,
            standardized_coefficients,
            original_scale_coefficients,
        )
    ]
    details.sort(
        key=lambda item: item["absolute_standardized_coefficient"], reverse=True
    )
    for rank, detail in enumerate(details, start=1):
        detail["absolute_contribution_rank"] = rank

    return {
        "interpretation": (
            "standardized_coefficientは他の特徴量を一定としたとき、特徴量が"
            "学習データ内で1標準偏差増加した場合の予測値の変化量です。"
            "絶対値が大きいほどRidgeモデル内での影響が大きく、正負は予測値を"
            "上げる・下げる方向を示します。因果関係を示す値ではありません。"
        ),
        "standardized_intercept": standardized_intercept,
        "original_scale_intercept": original_scale_intercept,
        "standardized_coefficients_by_feature": dict(
            zip(FEATURE_COLUMNS, standardized_coefficients)
        ),
        "original_scale_coefficients_by_feature": dict(
            zip(FEATURE_COLUMNS, original_scale_coefficients)
        ),
        "ranking_by_absolute_standardized_coefficient": details,
    }


def _unused_column_name(columns: Sequence[str], requested: str) -> str:
    name = requested
    while name in columns:
        name = f"model_{name}"
    return name


def _summarize_fold_metrics(
    fold_results: Sequence[Mapping[str, Any]], metric_group: str
) -> dict[str, dict[str, float]]:
    metric_names = ("r2", "rmse", "mae")
    values = {
        metric: [float(fold[metric_group][metric]) for fold in fold_results]
        for metric in metric_names
    }
    return {
        "mean": {metric: statistics.mean(items) for metric, items in values.items()},
        "std": {metric: statistics.pstdev(items) for metric, items in values.items()},
    }


def _build_cv_predictions(
    source_frame: Any,
    used_indices: Any,
    target_column: str,
    fold_assignments: Any,
    out_of_fold_predictions: Any,
    deps: Mapping[str, Any],
) -> Any:
    pd = deps["pd"]
    output = source_frame.loc[used_indices].copy()
    source_row_column = _unused_column_name(list(output.columns), "source_csv_row")
    fold_column = _unused_column_name(list(output.columns), "cv_fold")
    prediction_column = _unused_column_name(
        list(output.columns), f"predicted_{target_column}"
    )
    residual_column = _unused_column_name(
        list(output.columns), f"residual_{target_column}"
    )
    output.insert(0, source_row_column, output.index + 2)
    output[fold_column] = fold_assignments.loc[used_indices].astype(int)
    output[prediction_column] = out_of_fold_predictions.loc[used_indices]
    output[residual_column] = (
        pd.to_numeric(output[target_column], errors="coerce")
        - out_of_fold_predictions.loc[used_indices]
    )
    return output


def train_model(
    config: TrainingConfig | Mapping[str, Any],
    *,
    save_artifacts: bool = True,
) -> TrainingResult:
    """Nested CVで評価し、全データで再探索した最終モデルを返す。"""
    if not isinstance(config, TrainingConfig):
        config = TrainingConfig.from_mapping(config)

    deps = _import_dependencies()
    data = _prepare_data(
        config.input_path,
        config.target_column,
        config.encoding,
        deps,
    )
    X = data["X"]
    y = data["y"]
    if len(X) < config.n_splits * 2:
        raise ValueError(
            "各検証foldでR2を計算するには、データ行数がn_splitsの2倍以上必要です"
        )
    smallest_outer_train_size = len(X) - math.ceil(len(X) / config.n_splits)
    if smallest_outer_train_size < config.inner_splits:
        raise ValueError("外側foldの学習行数よりinner_splitsが大きいため探索できません")

    outer_cv = deps["KFold"](
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    pd = deps["pd"]
    out_of_fold_predictions = pd.Series(index=X.index, dtype="float64")
    fold_assignments = pd.Series(index=X.index, dtype="Int64")
    fold_results: list[dict[str, Any]] = []
    search_strategy = ""
    search_space: dict[str, Any] = {}

    for fold_number, (train_positions, validation_positions) in enumerate(
        outer_cv.split(X), start=1
    ):
        X_train = X.iloc[train_positions]
        X_validation = X.iloc[validation_positions]
        y_train = y.iloc[train_positions]
        y_validation = y.iloc[validation_positions]

        fold_pipeline, _, _ = _build_pipeline(
            config.model_name,
            deps,
            config.random_state,
            config.model_parameters,
        )
        fold_search, search_strategy, search_space = _build_hyperparameter_search(
            model_name=config.model_name,
            pipeline=fold_pipeline,
            deps=deps,
            inner_splits=config.inner_splits,
            random_state=config.random_state + fold_number,
            hyperparameter_grid=config.hyperparameter_grid,
        )
        # 探索に使うのは外側foldの学習データだけ。
        fold_search.fit(X_train, y_train)
        best_fold_pipeline = fold_search.best_estimator_
        train_predictions = best_fold_pipeline.predict(X_train)
        validation_predictions = best_fold_pipeline.predict(X_validation)
        out_of_fold_predictions.loc[X_validation.index] = validation_predictions
        fold_assignments.loc[X_validation.index] = fold_number
        fold_results.append(
            {
                "fold": fold_number,
                "train_rows": len(X_train),
                "validation_rows": len(X_validation),
                "train_metrics": _metric_values(y_train, train_predictions, deps),
                "validation_metrics": _metric_values(
                    y_validation, validation_predictions, deps
                ),
                "inner_cv_best_mae": float(-fold_search.best_score_),
                "best_parameters": _json_value(fold_search.best_params_),
            }
        )

    train_summary = _summarize_fold_metrics(fold_results, "train_metrics")
    validation_summary = _summarize_fold_metrics(fold_results, "validation_metrics")

    # 保存・予測用モデルはNested CV後に全データで改めて探索する。
    final_base_pipeline, _, package_version = _build_pipeline(
        config.model_name,
        deps,
        config.random_state,
        config.model_parameters,
    )
    final_search, search_strategy, search_space = _build_hyperparameter_search(
        model_name=config.model_name,
        pipeline=final_base_pipeline,
        deps=deps,
        inner_splits=config.inner_splits,
        random_state=config.random_state,
        hyperparameter_grid=config.hyperparameter_grid,
    )
    final_search.fit(X, y)
    pipeline = final_search.best_estimator_
    estimator = pipeline.named_steps["model"]

    metrics = {
        "model": config.model_name,
        "target_column": data["target_column"],
        "metrics_note": (
            f"train_metricsは外側{config.n_splits}foldの学習指標平均、"
            "test_metricsは外側foldの検証指標平均です。各外側foldの学習データ"
            "内だけでハイパーパラメータ探索を行っています。"
        ),
        "feature_count": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),
        "data": {
            "input_file": str(config.input_path.resolve()),
            "rows_read": data["rows_read"],
            "rows_used": data["rows_used"],
            "rows_excluded": data["rows_excluded"],
            "rows_with_invalid_smiles": data["rows_with_invalid_smiles"],
            "all_missing_features_filled_with_zero": data[
                "all_missing_features_filled_with_zero"
            ],
        },
        "cross_validation": {
            "method": "NestedKFold",
            "n_splits": config.n_splits,
            "inner_splits": config.inner_splits,
            "shuffle": True,
            "random_state": config.random_state,
            "train_metrics_summary": train_summary,
            "validation_metrics_summary": validation_summary,
            "fold_metrics": fold_results,
        },
        "train_metrics": train_summary["mean"],
        "test_metrics": validation_summary["mean"],
        "train_metrics_std": train_summary["std"],
        "test_metrics_std": validation_summary["std"],
        "hyperparameter_search": {
            "strategy": search_strategy,
            "scoring": "neg_mean_absolute_error",
            "candidate_count": math.prod(
                len(values) for values in search_space.values()
            ),
            "search_space": _json_value(search_space),
            "final_best_parameters": _json_value(final_search.best_params_),
            "final_inner_cv_best_mae": float(-final_search.best_score_),
            "final_candidate_results": _search_candidate_results(final_search),
        },
        "saved_model_training": (
            "full_dataset_after_nested_cross_validation_and_final_hyperparameter_search"
        ),
        "base_model_parameters": _json_value(config.model_parameters),
        "model_parameters": _json_value(estimator.get_params(deep=False)),
        "package_versions": {
            "model_package": package_version,
            "scikit_learn": deps["sklearn"].__version__,
            "pandas": deps["pd"].__version__,
        },
    }
    if config.model_name == "ridge":
        metrics["ridge_feature_contributions"] = _ridge_feature_contributions(pipeline)

    cv_predictions = _build_cv_predictions(
        data["frame"],
        X.index,
        data["target_column"],
        fold_assignments,
        out_of_fold_predictions,
        deps,
    )

    model_path: Path | None = None
    metrics_path: Path | None = None
    predictions_path: Path | None = None
    if save_artifacts:
        output_dir = config.resolved_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{config.model_name}_model.joblib"
        metrics_path = output_dir / f"{config.model_name}_metrics.json"
        predictions_path = output_dir / f"{config.model_name}_predictions.csv"
        deps["joblib"].dump(pipeline, model_path)
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cv_predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    return TrainingResult(
        config=config,
        pipeline=pipeline,
        metrics=metrics,
        cv_predictions=cv_predictions,
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )
