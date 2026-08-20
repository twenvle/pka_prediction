#!/usr/bin/env python3
"""XGBoost、SVR、Random Forest、Ridge用の共通回帰学習処理。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


# extract_smiles_features.py が生成する数値特徴量を全て使用する。
FEATURE_COLUMNS = (
    "MW",
    "logP",
    "TPSA",
    "HBA",
    "HBD",
    "rotatable_bonds",
    "aromatic_rings",
    "FractionCSP3",
    "formal_charge",
    "acid_type",
    "acid_group_count",
    "acid_center_aromatic_flag",
    "local_heteroatom_count",
    "local_halogen_count",
    "nearest_heteroatom_distance",
    "local_conjugation",
    "local_formal_charge",
)

MODEL_NAMES = ("xgboost", "svr", "random_forest", "ridge")


def _import_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import pandas as pd
        import sklearn
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import KFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
    except ImportError as exc:
        raise SystemExit(
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
        "KFold": KFold,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "SVR": SVR,
    }


def _build_pipeline(model_name: str, deps: Mapping[str, Any], random_state: int) -> Any:
    imputer = deps["SimpleImputer"](strategy="median")

    if model_name == "xgboost":
        try:
            import xgboost
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise SystemExit(
                "XGBoostがありません。次のコマンドで導入してください: pip install xgboost"
            ) from exc
        estimator = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [("imputer", imputer), ("model", estimator)]
        package_version = xgboost.__version__
    elif model_name == "svr":
        estimator = deps["SVR"](kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
        steps = [
            ("imputer", imputer),
            ("scaler", deps["StandardScaler"]()),
            ("model", estimator),
        ]
        package_version = deps["sklearn"].__version__
    elif model_name == "random_forest":
        estimator = deps["RandomForestRegressor"](
            n_estimators=500,
            max_features=1.0,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [("imputer", imputer), ("model", estimator)]
        package_version = deps["sklearn"].__version__
    elif model_name == "ridge":
        estimator = deps["Ridge"](alpha=1.0)
        steps = [
            ("imputer", imputer),
            ("scaler", deps["StandardScaler"]()),
            ("model", estimator),
        ]
        package_version = deps["sklearn"].__version__
    else:
        raise ValueError(f"未対応のモデルです: {model_name}")

    pipeline = deps["Pipeline"](steps)
    return pipeline, estimator, package_version


def _resolve_column(columns: Sequence[str], requested: str) -> str:
    matches = [column for column in columns if column.casefold() == requested.casefold()]
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
    if actual_target.casefold() in {column.casefold() for column in actual_features.values()}:
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
            "有効な学習データが少なすぎます。目的変数が数値である有効行を4行以上用意してください"
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


def _metric_values(y_true: Any, y_pred: Any, deps: Mapping[str, Any]) -> dict[str, float]:
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


def _ridge_feature_contributions(pipeline: Any) -> dict[str, Any]:
    """標準化空間と元スケールのRidge係数を返す。"""
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
                "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero"
            ),
        }
        for feature, coefficient, original_coefficient in zip(
            FEATURE_COLUMNS,
            standardized_coefficients,
            original_scale_coefficients,
        )
    ]
    details.sort(key=lambda item: item["absolute_standardized_coefficient"], reverse=True)
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


def _save_cv_predictions(
    source_frame: Any,
    used_indices: Any,
    target_column: str,
    fold_assignments: Any,
    out_of_fold_predictions: Any,
    output_path: Path,
    deps: Mapping[str, Any],
) -> None:
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
    output[residual_column] = pd.to_numeric(
        output[target_column], errors="coerce"
    ) - out_of_fold_predictions.loc[used_indices]
    output.to_csv(output_path, index=False, encoding="utf-8-sig")


def build_parser(model_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"全特徴量を使用して{model_name}回帰モデルを学習します。"
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="特徴量CSV")
    parser.add_argument(
        "-t", "--target-column", required=True, help="予測対象となる目的変数の列名"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("model_results") / model_name,
        help=f"保存先フォルダ（既定: model_results/{model_name}）",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="K-Foldの分割数（既定: 5）")
    parser.add_argument(
        "--random-state", type=int, default=42, help="乱数シード（既定: 42）"
    )
    parser.add_argument(
        "--encoding", default="utf-8-sig", help="入力CSVの文字コード（既定: utf-8-sig）"
    )
    return parser


def run_training(model_name: str, argv: Sequence[str] | None = None) -> int:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"未対応のモデルです: {model_name}")
    parser = build_parser(model_name)
    args = parser.parse_args(argv)
    if args.n_splits < 2:
        parser.error("--n-splits は2以上にしてください")

    deps = _import_dependencies()
    try:
        data = _prepare_data(args.input, args.target_column, args.encoding, deps)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    X = data["X"]
    y = data["y"]
    if len(X) < args.n_splits * 2:
        parser.error(
            "各検証foldでR2を計算するには、データ行数が --n-splits の2倍以上必要です"
        )

    cv = deps["KFold"](
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    pd = deps["pd"]
    out_of_fold_predictions = pd.Series(index=X.index, dtype="float64")
    fold_assignments = pd.Series(index=X.index, dtype="Int64")
    fold_results: list[dict[str, Any]] = []

    for fold_number, (train_positions, validation_positions) in enumerate(
        cv.split(X), start=1
    ):
        X_train = X.iloc[train_positions]
        X_validation = X.iloc[validation_positions]
        y_train = y.iloc[train_positions]
        y_validation = y.iloc[validation_positions]
        fold_pipeline, _, _ = _build_pipeline(
            model_name, deps, args.random_state
        )
        fold_pipeline.fit(X_train, y_train)
        train_predictions = fold_pipeline.predict(X_train)
        validation_predictions = fold_pipeline.predict(X_validation)
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
            }
        )

    train_summary = _summarize_fold_metrics(fold_results, "train_metrics")
    validation_summary = _summarize_fold_metrics(
        fold_results, "validation_metrics"
    )

    # 保存用モデルは交差検証後に全データで改めて学習する。
    pipeline, estimator, package_version = _build_pipeline(
        model_name, deps, args.random_state
    )
    pipeline.fit(X, y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{model_name}_model.joblib"
    metrics_path = args.output_dir / f"{model_name}_metrics.json"
    predictions_path = args.output_dir / f"{model_name}_predictions.csv"
    deps["joblib"].dump(pipeline, model_path)

    metrics = {
        "model": model_name,
        "target_column": data["target_column"],
        "metrics_note": (
            "train_metricsは5foldの学習指標平均、test_metricsは各foldの"
            "検証指標平均です。標準偏差とfold別結果はcross_validationにあります。"
        ),
        "feature_count": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),
        "data": {
            "input_file": str(args.input.resolve()),
            "rows_read": data["rows_read"],
            "rows_used": data["rows_used"],
            "rows_excluded": data["rows_excluded"],
            "rows_with_invalid_smiles": data["rows_with_invalid_smiles"],
            "all_missing_features_filled_with_zero": data[
                "all_missing_features_filled_with_zero"
            ],
        },
        "cross_validation": {
            "method": "KFold",
            "n_splits": args.n_splits,
            "shuffle": True,
            "random_state": args.random_state,
            "train_metrics_summary": train_summary,
            "validation_metrics_summary": validation_summary,
            "fold_metrics": fold_results,
        },
        # 従来との互換性のため、平均値を同じキーにも保存する。
        "train_metrics": train_summary["mean"],
        "test_metrics": validation_summary["mean"],
        "train_metrics_std": train_summary["std"],
        "test_metrics_std": validation_summary["std"],
        "saved_model_training": "full_dataset_after_cross_validation",
        "model_parameters": _json_value(estimator.get_params(deep=False)),
        "package_versions": {
            "model_package": package_version,
            "scikit_learn": deps["sklearn"].__version__,
            "pandas": deps["pd"].__version__,
        },
    }
    if model_name == "ridge":
        metrics["ridge_feature_contributions"] = _ridge_feature_contributions(
            pipeline
        )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_cv_predictions(
        data["frame"],
        X.index,
        data["target_column"],
        fold_assignments,
        out_of_fold_predictions,
        predictions_path,
        deps,
    )

    print(f"モデル: {model_path}")
    print(f"評価指標: {metrics_path}")
    print(f"予測値: {predictions_path}")
    print(
        f"{args.n_splits}-Fold CV評価: "
        f"R2={metrics['test_metrics']['r2']:.6f}, "
        f"RMSE={metrics['test_metrics']['rmse']:.6f}, "
        f"MAE={metrics['test_metrics']['mae']:.6f} "
        f"(std: R2={metrics['test_metrics_std']['r2']:.6f}, "
        f"RMSE={metrics['test_metrics_std']['rmse']:.6f}, "
        f"MAE={metrics['test_metrics_std']['mae']:.6f})"
    )
    return 0


if __name__ == "__main__":
    print(
        "このファイルは共通処理です。train_xgboost.py、train_svr.py、"
        "train_random_forest.py、train_ridge.py のいずれかを実行してください。",
        file=sys.stderr,
    )
    raise SystemExit(2)
