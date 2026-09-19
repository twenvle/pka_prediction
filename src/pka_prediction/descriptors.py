#!/usr/bin/env python3
"""SMILESから分子特徴量・酸性部位の特徴量を抽出する。

preprocessing.pyと同じ、Config / Result / DataFrame処理 / CSV処理の構成。
呼出し側のCLIやnotebookからimportして利用する。

使用例（pandasとRDKitが必要）:
    pip install pandas rdkit
    from descriptors_v2 import extract_descriptors, descriptors_dataframe
    row = extract_descriptors("CC(=O)O")
    output, exclusion_counts = descriptors_dataframe(df, smiles_column="smiles")

    from descriptors_v2 import DescriptorsConfig, descriptors_csv
    result = descriptors_csv(DescriptorsConfig("molecules.csv", "result.csv"))
    result.dataframe

descriptors_csvはヘッダー付きのカンマ区切りCSVをpandasで読み書きする。
SMILES列名は前後の空白と大文字・小文字を無視して一意に解決する。
表の入力値は文字列として保持し、IDの先頭ゼロや文字列NAを変換しない。

errors="keep"（既定）は欠損・不正なSMILESをerror列付きで保持する。
errors="drop"はそれらを除外し、errors="raise"はValueErrorで停止する。
元のファイル読込では空のSMILESをスキップしていたが、本版は上記方針に従う。
DataFrame APIは元のindex、入力列の型、行順を保持し、入力を変更しない。
重複indexは扱えるが、重複列名・文字列以外の列名は拒否する。
数値の欠損は単一行APIではNone、DataFrameではnullable数値型、CSVでは空欄。

特徴量の定義は元の実装を維持する。酸種はカルボン酸=0、スルホン酸=1、
対象酸なし=欠損。両方ある場合は1を優先し、局所特徴量は全部位で合計する。
酸性中心はSMARTSの:1の酸素で、芳香族フラグもこの酸素自身を調べる。
最近接ヘテロ原子には酸官能基自身の原子を含む。部位の近傍が重なる場合も
部位ごとに数えるため、合計には重複があり、合計したフラグは0/1とは限らない。

CSVの出力は既定でUTF-8 BOM付き。
出力親ディレクトリを作成する。入出力に同じファイルは指定できない。
バッチ処理は結果をメモリ上に保持するため、巨大な入力は分割して処理する。
"""

from __future__ import annotations

import codecs
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "DescriptorsConfig",
    "DescriptorsResult",
    "FEATURE_COLUMNS",
    "EXCLUSION_REASONS",
    "extract_descriptors",
    "descriptors_dataframe",
    "descriptors_csv",
]

ACID_SMARTS: tuple[tuple[int, str], ...] = (
    (1, "[O;H1,-1:1]-[S;X4](=[O;X1])(=[O;X1])"),
    (0, "[O;H1,-1:1]-[C;X3](=[O;X1])"),
)
HETERO_ATOMIC_NUMBERS = frozenset({7, 8, 15, 16, 34})
HALOGEN_ATOMIC_NUMBERS = frozenset({9, 17, 35, 53})
LOCAL_RADIUS = 2

FEATURE_COLUMNS = (
    "canonical_smiles",
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
    "error",
)
SITE_COLUMNS = (
    "acid_center_aromatic_flag",
    "local_heteroatom_count",
    "local_halogen_count",
    "nearest_heteroatom_distance",
    "local_conjugation",
    "local_formal_charge",
)
_FLOAT_COLUMNS = frozenset({"MW", "logP", "TPSA", "FractionCSP3"})
_TEXT_COLUMNS = frozenset({"canonical_smiles", "error"})
_ERROR_POLICIES = frozenset({"keep", "drop", "raise"})
EXCLUSION_REASONS = ("missing_smiles", "invalid_smiles")


def _validate_smiles_column(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("smiles_columnには空でない文字列を指定してください")
    return value.strip()


def _validate_errors(value: object) -> None:
    if not isinstance(value, str) or value not in _ERROR_POLICIES:
        raise ValueError("errorsにはkeep、drop、raiseのいずれかを指定してください")


def _same_file(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    return first.exists() and second.exists() and first.samefile(second)


@dataclass(frozen=True)
class DescriptorsConfig:
    """CSV特徴量抽出の入出力条件。"""

    input_path: Path | str
    output_path: Path | str
    name_column: str = "name"
    smiles_column: str = "smiles"
    encoding: str = "utf-8-sig"
    output_encoding: str = "utf-8-sig"
    errors: str = "keep"

    def __post_init__(self) -> None:
        _validate_errors(self.errors)
        codecs.lookup(self.encoding)
        codecs.lookup(self.output_encoding)
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_path", Path(self.output_path))
        if _same_file(self.input_path, self.output_path):
            raise ValueError(
                "入力ファイルと出力ファイルには異なるパスを指定してください"
            )


@dataclass
class DescriptorsResult:
    """特徴量抽出後のDataFrameと除外件数。

    rows_readはヘッダー・完全な空行を除く読込レコード数。
    rows_writtenはCSVへ出力した行数。exclusion_countsは実際の除外理由別件数。
    """

    config: DescriptorsConfig
    dataframe: Any
    rows_read: int
    rows_written: int
    exclusion_counts: dict[str, int]

    @property
    def rows_excluded(self) -> int:
        return self.rows_read - self.rows_written


def _compile_queries(Chem: Any) -> tuple[tuple[int, Any, int], ...]:
    queries: list[tuple[int, Any, int]] = []
    for acid_type, smarts in ACID_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise RuntimeError(f"内部SMARTSが不正です: {acid_type}: {smarts}")
        mapped = [
            atom.GetIdx() for atom in query.GetAtoms() if atom.GetAtomMapNum() == 1
        ]
        if len(mapped) != 1:
            raise RuntimeError(f":1の酸性中心は1個必要です: {smarts}")
        queries.append((acid_type, query, mapped[0]))
    return tuple(queries)


def _import_dependencies() -> dict[str, Any]:
    """preprocessing.pyと同様、処理を呼び出した時だけ依存関係を読み込む。"""
    try:
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    except ImportError as exc:
        raise RuntimeError(
            "特徴量抽出にはpandasとRDKitが必要です。次のコマンドで導入してください:\n"
            "pip install pandas rdkit"
        ) from exc
    return {
        "pd": pd,
        "Chem": Chem,
        "Descriptors": Descriptors,
        "Crippen": Crippen,
        "rdMolDescriptors": rdMolDescriptors,
    }


def _distances_from(mol: Any, start_idx: int) -> list[int | None]:
    distances: list[int | None] = [None] * mol.GetNumAtoms()
    distances[start_idx] = 0
    queue: deque[int] = deque([start_idx])
    while queue:
        atom_idx = queue.popleft()
        distance = distances[atom_idx]
        assert distance is not None
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if distances[neighbor_idx] is None:
                distances[neighbor_idx] = distance + 1
                queue.append(neighbor_idx)
    return distances


def _find_acid_sites(
    mol: Any,
    queries: Sequence[tuple[int, Any, int]],
) -> list[tuple[int, int]]:
    sites: dict[int, int] = {}
    for acid_type, query, center_query_idx in queries:
        for match in mol.GetSubstructMatches(query, uniquify=True):
            sites.setdefault(match[center_query_idx], acid_type)
    return sorted(sites.items())


def _molecule_descriptors(
    mol: Any,
    Chem: Any,
    Descriptors: Any,
    Crippen: Any,
    rdMolDescriptors: Any,
) -> dict[str, int | float]:
    return {
        "MW": round(Descriptors.MolWt(mol), 6),
        "logP": round(Crippen.MolLogP(mol), 6),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 6),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": round(rdMolDescriptors.CalcFractionCSP3(mol), 6),
        "formal_charge": Chem.GetFormalCharge(mol),
    }


def _site_descriptors(mol: Any, center_idx: int) -> dict[str, int | str]:
    center = mol.GetAtomWithIdx(center_idx)
    distances = _distances_from(mol, center_idx)
    local_indices = {
        idx
        for idx, distance in enumerate(distances)
        if distance is not None and distance <= LOCAL_RADIUS
    }
    local_heteroatom_count = sum(
        mol.GetAtomWithIdx(idx).GetAtomicNum() in HETERO_ATOMIC_NUMBERS
        for idx in local_indices
        if idx != center_idx
    )
    local_halogen_count = sum(
        mol.GetAtomWithIdx(idx).GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS
        for idx in local_indices
    )
    other_hetero_distances = [
        distance
        for idx, distance in enumerate(distances)
        if idx != center_idx
        and distance is not None
        and mol.GetAtomWithIdx(idx).GetAtomicNum() in HETERO_ATOMIC_NUMBERS
    ]
    local_conjugation = int(
        any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in local_indices)
        or any(
            bond.GetIsConjugated()
            and bond.GetBeginAtomIdx() in local_indices
            and bond.GetEndAtomIdx() in local_indices
            for bond in mol.GetBonds()
        )
    )
    return {
        "acid_center_aromatic_flag": int(center.GetIsAromatic()),
        "local_heteroatom_count": local_heteroatom_count,
        "local_halogen_count": local_halogen_count,
        "nearest_heteroatom_distance": (
            min(other_hetero_distances) if other_hetero_distances else ""
        ),
        "local_conjugation": local_conjugation,
        "local_formal_charge": sum(
            mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in local_indices
        ),
    }


def extract_descriptors(
    smiles: object,
    *,
    dependencies: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """1個のSMILESを抽出する。不正入力はerror列に理由を返す。

    Chemなどの引数は不要。依存関係や内部計算の問題は例外として通知する。
    """
    result: dict[str, object] = {
        column: "" if column in _TEXT_COLUMNS else None for column in FEATURE_COLUMNS
    }
    result["acid_group_count"] = 0
    if not isinstance(smiles, str):
        result["error"] = "Missing or non-string SMILES"
        return result
    smiles = smiles.strip()
    if not smiles:
        result["error"] = "Missing SMILES"
        return result

    deps = dict(_import_dependencies() if dependencies is None else dependencies)
    Chem = deps["Chem"]
    try:
        mol = Chem.MolFromSmiles(smiles)
    except (RuntimeError, ValueError):
        mol = None
    if mol is None or mol.GetNumAtoms() == 0:
        result["error"] = "Invalid SMILES"
        return result
    result.update(
        {
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            **_molecule_descriptors(
                mol,
                Chem,
                deps["Descriptors"],
                deps["Crippen"],
                deps["rdMolDescriptors"],
            ),
        }
    )
    queries = deps["queries"] if "queries" in deps else _compile_queries(Chem)
    sites = _find_acid_sites(mol, queries)
    if not sites:
        result.update({column: 0 for column in SITE_COLUMNS})
        return result

    site_descriptors = [_site_descriptors(mol, center_idx) for center_idx, _ in sites]
    result.update(
        {
            "acid_type": max(acid_type for _, acid_type in sites),
            "acid_group_count": len(sites),
            **{
                column: sum(
                    int(site[column]) if site[column] != "" else 0
                    for site in site_descriptors
                )
                for column in SITE_COLUMNS
            },
        }
    )
    return result


def _validate_columns(columns: Sequence[object]) -> None:
    if any(not isinstance(column, str) for column in columns):
        raise ValueError("入力の列名はすべて文字列にしてください")
    if len(set(columns)) != len(columns):
        raise ValueError("入力に重複した列名があります")


def _resolve_column(columns: Sequence[str], requested: str) -> str:
    requested = _validate_smiles_column(requested)
    matches = [
        column
        for column in columns
        if column.strip().casefold() == requested.casefold()
    ]
    if not matches:
        raise ValueError(f"必須列'{requested}'が入力にありません")
    if len(matches) > 1:
        raise ValueError(f"空白・大文字小文字だけが異なる重複列があります: {requested}")
    return matches[0]


def _input_column_mapping(columns: Sequence[str]) -> dict[str, str]:
    """計算列との衝突を回避する。元の列順にinput_を必要回数だけ付ける。"""
    used = set(FEATURE_COLUMNS)
    mapping: dict[str, str] = {}
    for column in columns:
        output_column = column
        while output_column in used:
            output_column = f"input_{output_column}"
        mapping[column] = output_column
        used.add(output_column)
    return mapping


def descriptors_dataframe(
    frame: Any,
    *,
    smiles_column: str = "smiles",
    errors: str = "keep",
    dependencies: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, int]]:
    """特徴量付きDataFrameと除外理由別件数を返す。

    preprocess_dataframeと同様にdependenciesを受け取る。
    元の入力列・型・indexを保持し、入力を変更しない。空のDataFrameも扱う。
    計算列と同名の入力列はinput_付きに改名する。
    errors="keep"ではエラー行を保持し、除外件数はすべて0になる。
    errors="drop"では除外理由をmissing_smiles / invalid_smilesで数える。
    """
    _validate_errors(errors)
    deps = dict(_import_dependencies() if dependencies is None else dependencies)
    pd = deps["pd"]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frameにはpandas.DataFrameを渡してください")
    columns = list(frame.columns)
    _validate_columns(columns)
    actual_smiles_column = _resolve_column(columns, smiles_column)
    rows: list[dict[str, object]] = []
    positions: list[int] = []
    exclusion_counts: Counter[str] = Counter(
        {reason: 0 for reason in EXCLUSION_REASONS}
    )
    # バッチで使用するSMARTSは一度だけコンパイルし、各行へ同じ依存関係を渡す。
    if not frame.empty and "queries" not in deps:
        deps["queries"] = _compile_queries(deps["Chem"])
    for position, value in enumerate(frame[actual_smiles_column].tolist()):
        descriptors = extract_descriptors(value, dependencies=deps)
        reason = str(descriptors["error"])
        if reason:
            if errors == "raise":
                raise ValueError(
                    f"行{position + 1}（index={frame.index[position]!r}）: {reason}"
                )
            if errors == "drop":
                exclusion_reason = (
                    "invalid_smiles" if reason == "Invalid SMILES" else "missing_smiles"
                )
                exclusion_counts[exclusion_reason] += 1
                continue
        positions.append(position)
        rows.append(descriptors)

    # 入力列をコピーして型とindexを保持し、計算列はラベルではなく位置で代入する。
    output = frame.iloc[positions].copy()
    mapping = _input_column_mapping(columns)
    output.columns = [mapping[column] for column in columns]
    feature_frame = pd.DataFrame.from_records(rows, columns=FEATURE_COLUMNS)
    for column in FEATURE_COLUMNS:
        dtype = (
            "string"
            if column in _TEXT_COLUMNS
            else ("Float64" if column in _FLOAT_COLUMNS else "Int64")
        )
        output[column] = feature_frame[column].astype(dtype).array
    return output, dict(exclusion_counts)


def descriptors_csv(config: DescriptorsConfig | Mapping[str, Any]) -> DescriptorsResult:
    """CSVを読み込み、特徴量付きの新しいCSVを保存する。"""
    if not isinstance(config, DescriptorsConfig):
        if not isinstance(config, Mapping):
            raise TypeError("configにはDescriptorsConfigまたはMappingを渡してください")
        config = DescriptorsConfig(**dict(config))
    deps = _import_dependencies()
    pd = deps["pd"]
    frame = pd.read_csv(
        config.input_path,
        encoding=config.encoding,
        dtype=str,
        keep_default_na=False,
    )
    output, exclusion_counts = descriptors_dataframe(
        frame,
        smiles_column=config.smiles_column,
        errors=config.errors,
        dependencies=deps,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        config.output_path,
        index=False,
        encoding=config.output_encoding,
    )
    return DescriptorsResult(
        config=config,
        dataframe=output,
        rows_read=len(frame),
        rows_written=len(output),
        exclusion_counts=exclusion_counts,
    )
