#!/usr/bin/env python3
"""CSV中のSMILESを検査し、学習に使用できる行だけを保存する前処理。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_COLUMNS = ("name", "smiles")

# アルカリ金属、アルカリ土類金属、遷移金属、ランタノイド、
# アクチノイド、および代表的なポスト遷移金属。
METAL_ATOMIC_NUMBERS = frozenset(
    {
        3,
        4,
        11,
        12,
        13,
        19,
        20,
        31,
        37,
        38,
        49,
        50,
        55,
        56,
        81,
        82,
        83,
        84,
        87,
        88,
        113,
        114,
        115,
        116,
    }
    | set(range(21, 31))
    | set(range(39, 49))
    | set(range(57, 81))
    | set(range(89, 113))
)

EXCLUSION_REASONS = (
    "missing_smiles",
    "invalid_smiles",
    "contains_metal",
    "ionic_smiles",
    "3d_generation_failed",
    "missing_values",
)


@dataclass(frozen=True)
class PreprocessingConfig:
    """CSV前処理の入出力とSMILES検査条件。"""

    input_path: Path | str
    output_path: Path | str
    name_column: str = "name"
    smiles_column: str = "smiles"
    encoding: str = "utf-8-sig"
    output_encoding: str = "utf-8-sig"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not str(self.name_column).strip():
            raise ValueError("name_columnを指定してください")
        if not str(self.smiles_column).strip():
            raise ValueError("smiles_columnを指定してください")
        if not 0 <= self.random_seed <= 2_147_483_646:
            raise ValueError("random_seedは0以上2147483646以下にしてください")

        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_path", Path(self.output_path))


@dataclass
class PreprocessingResult:
    """前処理後のDataFrameと除外件数。"""

    config: PreprocessingConfig
    dataframe: Any
    rows_read: int
    rows_written: int
    exclusion_counts: dict[str, int]

    @property
    def rows_excluded(self) -> int:
        return self.rows_read - self.rows_written


def _import_dependencies() -> dict[str, Any]:
    try:
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError(
            "前処理にはpandasとRDKitが必要です。次のコマンドで導入してください:\n"
            "pip install pandas rdkit"
        ) from exc
    return {"pd": pd, "Chem": Chem, "AllChem": AllChem}


def _resolve_column(columns: Sequence[str], requested: str) -> str:
    """大文字・小文字を区別せず、要求された列を一意に解決する。"""
    matches = [
        column for column in columns if str(column).casefold() == requested.casefold()
    ]
    if not matches:
        raise ValueError(f"必須列'{requested}'が入力CSVにありません")
    if len(matches) > 1:
        raise ValueError(f"大文字・小文字だけが異なる重複列があります: {requested}")
    return str(matches[0])


def _normalize_smiles(value: Any, pd: Any) -> str | None:
    if pd.isna(value):
        return None
    smiles = str(value).strip()
    return smiles or None


def _contains_metal(mol: Any) -> bool:
    return any(
        atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS for atom in mol.GetAtoms()
    )


def _is_ionic(mol: Any) -> bool:
    """正味電荷0の双性イオンも含め、形式電荷を持つ分子をイオンとする。"""
    return any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())


def _embedding_parameters(AllChem: Any, random_seed: int, use_random: bool) -> Any:
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = random_seed
    parameters.enforceChirality = True
    parameters.useRandomCoords = use_random
    return parameters


def _can_generate_3d(mol: Any, Chem: Any, AllChem: Any, random_seed: int) -> bool:
    """ETKDGで3D配座を作る。通常座標で失敗した場合はランダム座標で再試行する。"""
    seeds = (random_seed, random_seed + 1)
    for attempt, seed in enumerate(seeds):
        try:
            molecule_3d = Chem.AddHs(Chem.Mol(mol))
            parameters = _embedding_parameters(
                AllChem,
                seed,
                use_random=attempt == 1,
            )
            status = AllChem.EmbedMolecule(molecule_3d, parameters)
            if status == 0 and molecule_3d.GetNumConformers() > 0:
                return True
        except (RuntimeError, ValueError):
            continue
    return False


def _molecule_exclusion_reason(
    smiles_value: Any,
    *,
    pd: Any,
    Chem: Any,
    AllChem: Any,
    random_seed: int,
) -> str | None:
    smiles = _normalize_smiles(smiles_value, pd)
    if smiles is None:
        return "missing_smiles"

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except (RuntimeError, ValueError):
        mol = None
    # ダミー原子（*）を含むSMILESも、実体のある構造ではないものとして除外する。
    if mol is None or mol.GetNumAtoms() == 0:
        return "invalid_smiles"
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return "invalid_smiles"
    if _contains_metal(mol):
        return "contains_metal"
    if _is_ionic(mol):
        return "ionic_smiles"
    if not _can_generate_3d(mol, Chem, AllChem, random_seed):
        return "3d_generation_failed"
    return None


def _replace_blank_strings_with_missing(frame: Any, pd: Any) -> Any:
    output = frame.copy()
    text_columns = output.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        output[column] = output[column].replace(r"^\s*$", pd.NA, regex=True)
    return output


def preprocess_dataframe(
    frame: Any,
    *,
    name_column: str = "name",
    smiles_column: str = "smiles",
    random_seed: int = 42,
    dependencies: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, int]]:
    """DataFrameを検査し、採用行と除外理由別件数を返す。"""
    deps = dict(dependencies or _import_dependencies())
    pd = deps["pd"]
    Chem = deps["Chem"]
    AllChem = deps["AllChem"]

    if frame.empty:
        raise ValueError("入力CSVにデータ行がありません")
    actual_name_column = _resolve_column(list(frame.columns), name_column)
    actual_smiles_column = _resolve_column(list(frame.columns), smiles_column)
    if actual_name_column == actual_smiles_column:
        raise ValueError("name列とsmiles列は別の列にしてください")

    counts: Counter[str] = Counter({reason: 0 for reason in EXCLUSION_REASONS})
    kept_indices: list[Any] = []
    for index, smiles_value in frame[actual_smiles_column].items():
        reason = _molecule_exclusion_reason(
            smiles_value,
            pd=pd,
            Chem=Chem,
            AllChem=AllChem,
            random_seed=random_seed,
        )
        if reason is None:
            kept_indices.append(index)
        else:
            counts[reason] += 1

    structurally_valid = frame.loc[kept_indices].copy()
    structurally_valid = _replace_blank_strings_with_missing(structurally_valid, pd)
    complete_rows = structurally_valid.dropna(axis=0, how="any").copy()
    counts["missing_values"] += len(structurally_valid) - len(complete_rows)
    complete_rows.reset_index(drop=True, inplace=True)
    return complete_rows, dict(counts)


def preprocess_csv(
    config: PreprocessingConfig | Mapping[str, Any],
) -> PreprocessingResult:
    """CSVを読み込み、条件を満たす完全な行だけを新しいCSVへ保存する。"""
    if not isinstance(config, PreprocessingConfig):
        config = PreprocessingConfig(**dict(config))

    deps = _import_dependencies()
    pd = deps["pd"]
    frame = pd.read_csv(config.input_path, encoding=config.encoding)
    cleaned, exclusion_counts = preprocess_dataframe(
        frame,
        name_column=config.name_column,
        smiles_column=config.smiles_column,
        random_seed=config.random_seed,
        dependencies=deps,
    )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(
        config.output_path,
        index=False,
        encoding=config.output_encoding,
    )
    return PreprocessingResult(
        config=config,
        dataframe=cleaned,
        rows_read=len(frame),
        rows_written=len(cleaned),
        exclusion_counts=exclusion_counts,
    )
