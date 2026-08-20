#!/usr/bin/env python3
"""SMILESから分子特徴量と酸性部位の特徴量を1分子1行でCSV出力する。

必要パッケージ:
    pip install rdkit

使用例:
    python extract_smiles_features.py "CC(=O)O"
    python extract_smiles_features.py -i molecules.csv -o result.csv

入力CSV/TSVの全列を出力へ引き継ぐ。複数の酸性部位がある場合でも
出力は1分子につき1行で、局所特徴量は全部位の値を合計して格納する。
acid_type はカルボン酸=0、スルホン酸=1、対象酸なし=空欄とする。
両方の酸を持つ場合はスルホン酸を優先して1とする。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, TextIO

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
except ImportError as exc:
    raise SystemExit(
        "RDKitが必要です。次のコマンドで導入してください: pip install rdkit"
    ) from exc


# :1 は酸性プロトンを失う原子。酸種コードはカルボン酸=0、スルホン酸=1。
ACID_SMARTS: tuple[tuple[int, str], ...] = (
    (1, "[O;H1,-1:1]-[S;X4](=[O;X1])(=[O;X1])"),
    (0, "[O;H1,-1:1]-[C;X3](=[O;X1])"),
)

HETERO_ATOMIC_NUMBERS = frozenset({7, 8, 15, 16, 34})  # N, O, P, S, Se
HALOGEN_ATOMIC_NUMBERS = frozenset({9, 17, 35, 53})  # F, Cl, Br, I
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


def _compile_queries() -> tuple[tuple[int, Chem.Mol, int], ...]:
    queries: list[tuple[int, Chem.Mol, int]] = []
    for acid_type, smarts in ACID_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise RuntimeError(f"内部SMARTSが不正です: {acid_type}: {smarts}")
        mapped = [
            atom.GetIdx() for atom in query.GetAtoms() if atom.GetAtomMapNum() == 1
        ]
        if len(mapped) != 1:
            raise RuntimeError(f":1 の酸性中心は1個必要です: {smarts}")
        queries.append((acid_type, query, mapped[0]))
    return tuple(queries)


ACID_QUERIES = _compile_queries()


def _distances_from(mol: Chem.Mol, start_idx: int) -> list[int | None]:
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


def _find_acid_sites(mol: Chem.Mol) -> list[tuple[int, int]]:
    sites: dict[int, int] = {}
    for acid_type, query, center_query_idx in ACID_QUERIES:
        for match in mol.GetSubstructMatches(query, uniquify=True):
            sites.setdefault(match[center_query_idx], acid_type)
    return sorted(sites.items())


def _molecule_features(mol: Chem.Mol) -> dict[str, int | float]:
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


def _site_features(mol: Chem.Mol, center_idx: int) -> dict[str, int | str]:
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


def extract_features(smiles: str) -> dict[str, object]:
    """1個のSMILESから、出力1行分の特徴量を返す。"""
    result: dict[str, object] = {column: "" for column in FEATURE_COLUMNS}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result.update(
            {
                "acid_type": "",
                "acid_group_count": 0,
                "error": "Invalid SMILES",
            }
        )
        return result

    result.update(
        {
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            **_molecule_features(mol),
            "error": "",
        }
    )
    sites = _find_acid_sites(mol)
    if not sites:
        result.update(
            {
                "acid_type": "",
                "acid_group_count": 0,
                **{column: 0 for column in SITE_COLUMNS},
            }
        )
        return result

    site_features = [_site_features(mol, center_idx) for center_idx, _ in sites]
    result.update(
        {
            # 両方ある場合はスルホン酸(1)を優先する。
            "acid_type": max(acid_type for _, acid_type in sites),
            "acid_group_count": len(sites),
            **{
                column: sum(
                    int(site[column]) if site[column] != "" else 0
                    for site in site_features
                )
                for column in SITE_COLUMNS
            },
        }
    )
    return result


def _read_input_file(
    path: Path, smiles_column: str
) -> Iterator[tuple[dict[str, str], str]]:
    if path.suffix.lower() in {".smi", ".smiles", ".txt"}:
        with path.open("r", encoding="utf-8-sig") as handle:
            for row_number, line in enumerate(handle, start=1):
                fields = line.strip().split(maxsplit=1)
                if not fields or fields[0].startswith("#"):
                    continue
                input_id = fields[1] if len(fields) == 2 else str(row_number)
                yield {"input_id": input_id, "input_smiles": fields[0]}, fields[0]
        return

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        if path.suffix.lower() == ".tsv":
            dialect = csv.excel_tab
        else:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            except csv.Error:
                dialect = csv.excel
        rows = [
            row
            for row in csv.reader(handle, dialect=dialect)
            if row and any(cell.strip() for cell in row)
        ]

    if not rows:
        return
    header = [cell.strip() for cell in rows[0]]
    lookup = {name.casefold(): idx for idx, name in enumerate(header)}
    requested = smiles_column.casefold()
    if requested not in lookup:
        raise ValueError(
            f"SMILES列 '{smiles_column}' が入力ファイルに見つかりません"
        )
    smiles_idx = lookup[requested]

    for row in rows[1:]:
        if smiles_idx >= len(row) or not row[smiles_idx].strip():
            continue
        input_record = {
            column: (row[idx].strip() if idx < len(row) else "")
            for idx, column in enumerate(header)
        }
        yield input_record, row[smiles_idx].strip()


def _iter_inputs(
    args: argparse.Namespace,
) -> Iterable[tuple[dict[str, str], str]]:
    for index, smiles in enumerate(args.smiles, start=1):
        yield {"input_id": str(index), "input_smiles": smiles}, smiles
    if args.input:
        yield from _read_input_file(args.input, args.smiles_column)
    if not args.smiles and not args.input and not sys.stdin.isatty():
        for index, line in enumerate(sys.stdin, start=1):
            fields = line.strip().split(maxsplit=1)
            if fields and not fields[0].startswith("#"):
                input_id = fields[1] if len(fields) == 2 else str(index)
                yield {"input_id": input_id, "input_smiles": fields[0]}, fields[0]


def _merge_input_and_features(
    input_record: Mapping[str, object], features: Mapping[str, object]
) -> dict[str, object]:
    """入力列を保存し、計算列と同名なら入力列側に input_ を付ける。"""
    merged: dict[str, object] = {}
    for column, value in input_record.items():
        output_column = column
        while output_column in features or output_column in merged:
            output_column = f"input_{output_column}"
        merged[output_column] = value
    merged.update(features)
    return merged


def _write_csv(rows: Iterable[Mapping[str, object]], handle: TextIO) -> int:
    materialized = list(rows)
    fieldnames: list[str] = []
    for row in materialized:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in materialized:
        writer.writerow({column: row.get(column, "") for column in fieldnames})
    return len(materialized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SMILESから分子特徴量と酸性部位の局所特徴量を抽出します。"
    )
    parser.add_argument("smiles", nargs="*", help="1個以上のSMILES")
    parser.add_argument("-i", "--input", type=Path, help="CSV/TSV/SMI入力ファイル")
    parser.add_argument("-o", "--output", type=Path, help="出力CSV（省略時は画面）")
    parser.add_argument(
        "--smiles-column", default="SMILES", help="SMILES列名（既定: SMILES）"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        records = list(_iter_inputs(args))
    except ValueError as exc:
        parser.error(str(exc))
    if not records:
        parser.error("SMILES、--input、または標準入力を指定してください")

    output_rows = (
        _merge_input_and_features(input_record, extract_features(smiles))
        for input_record, smiles in records
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            count = _write_csv(output_rows, handle)
        print(f"{count}行を {args.output} に保存しました", file=sys.stderr)
    else:
        _write_csv(output_rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
