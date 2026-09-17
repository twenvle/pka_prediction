from dataclasses import dataclass
from pathlib import Path
from typing import Any

FEATURE_COLUMNS = [
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
]


@dataclass(frozen=True)
class PredictConfig:
    input_path: Path | str
    output_path: Path | str
    model_path: Path | str
    encoding: str = "utf-8-sig"
    output_encoding: str = "utf-8-sig"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_path", Path(self.output_path))


@dataclass
class PredictResult:
    config: PredictConfig
    dataframe: Any


def _import_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "予測にはpandas, joblibが必要です。次のコマンドで導入してください:\n"
            "pip install pandas joblib"
        ) from exc
    return {"pd": pd, "joblib": joblib}


def predict_pka(config: PredictConfig) -> PredictResult:
    dependencies = _import_dependencies()
    pd = dependencies["pd"]
    joblib = dependencies["joblib"]

    # 学習済みモデルを読み込む
    model = joblib.load(config.model_path)

    # 予測対象CSVを読み込む
    df = pd.read_csv(config.input_path, encoding=config.encoding)

    # 特徴量を取り出す
    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # pKaを予測
    predicted_pka = model.predict(X)

    # DataFrameに追加
    df["predicted_pKa"] = predicted_pka

    # 保存
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.output_path, index=False, encoding=config.output_encoding)

    return PredictResult(config=config, dataframe=df)
