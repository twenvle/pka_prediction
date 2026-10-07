#!/usr/bin/env python3
"""安息香酸誘導体のSMILESから30個の特徴量を抽出する。

元の descriptors.py の Config/Result、独自例外、遅延import、1分子単位の
calculate_descriptors、行単位処理、valid/invalid CSV、on_error方針を踏襲する。
GaussianログやDFT計算は不要。依存パッケージはRDKitとpandasのみ。

対象と定義:
* 連結した1分子中の、6員の全炭素芳香環に直接結合する中性COOHを
  すべて対象とする。各基準環の6結合はすべて芳香族結合であること。
  脂肪族COOHは酸点の選択対象から外す。対象COOHが0個ならエラー。
* 基準環に縮合した環も扱い、環外の原子に接続する各位置を置換位置
  と数える。ナフタレンの縮合部は通常2位置。非芳香族の橋も同様。
* 複数の対象COOHがある場合、各酸点の構造記述子と局所記述子を
  等重みで算術平均する。全分子記述子とHalogenAtomCountは平均しない。
  脂肪族COOHも分子中に保持するため全分子記述子や置換基効果には寄与。
* Total/Ortho/Meta/ParaSubstituentCount は基準COOHを除く置換「位置」数。
  他のCOOHは置換基として扱う。平均値なので整数とは限らない。
  未置換安息香酸はすべて0。
* HeteroatomsOutsideCOOH は各基準COOHの2個のOだけを除くH/C以外の
  原子数の平均。他のCOOHのO（脂肪族COOHも含む）は数える。
* AdditionalAromaticRingCount は各酸点でRDKitの芳香環数から基準環
  1個を引いた値の平均。SubstituentSP3CarbonCount は各基準環/COOH外
  の全sp3炭素数の平均。
* Gasteiger電荷は全Hを明示化した分子から取得し、酸性Hを個別に特定。
  EStateは水素を省略した分子上で計算する。重原子の局所対応は同じ分子
  の原子番号に基づき、H追加時には原子プロパティで対応を検証する。
* 芳香環直結のカルボキシラートは酸性Hがないためエラー。全炭素6員
  芳香環に対応できない芳香環直結COOH、基準環が曖昧な酸点もエラー。
  酸点の一部を任意に選んだり塩を中和したりしない。ラジカル、ダミー原子、
  除去できない明示H（同位体標識Hなど）、未知Gasteigerパラメータ、
  NaN/infは明示的エラーにする。

使用例（CSVに name,smiles 列。その他の列もそのまま保持）:
    python benzoic_acid_descriptors.py input.csv valid.csv
    python benzoic_acid_descriptors.py input.csv valid.csv --on-error raise

Python API:
    config = DescriptorExtractionConfig("input.csv", "valid.csv")
    result = extract_descriptors(config)
    features = calculate_descriptors("O=C(O)c1ccccc1")

CSVは文字列として読み、IDの先頭ゼロと文字列NAを保持する。
invalid出力の既定名は valid_invalid.csv。source_indexは入力データ行の
0-based番号（ヘッダーや空行を含む物理行番号ではない）。continueでは
失敗行を分離し、raiseでは最初の失敗で停止して出力CSVを書き込まない。
ヘッダーのみの入力にも、両CSVのヘッダーを出力する。

API参考:
https://www.rdkit.org/docs/source/rdkit.Chem.EState.EState.html
https://www.rdkit.org/docs/source/rdkit.Chem.rdPartialCharges.html
"""

from __future__ import annotations

import argparse
import codecs
import csv
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "DescriptorExtractionConfig",
    "DescriptorExtractionResult",
    "DescriptorExtractionError",
    "DESCRIPTOR_COLUMNS",
    "calculate_descriptors",
    "extract_descriptors",
]

# SMARTSの原子map番号: 1=COOH炭素, 2=C=O酸素, 3=OH/カルボキシラート酸素。
ACID_SMARTS = "[C;X3;+0:1](=[O;X1;+0:2])-[O;X2;H1;+0:3]"
CARBOXYLATE_SMARTS = "[C;X3;+0:1](=[O;X1;+0:2])-[O;X1;-1:3]"

DESCRIPTOR_COLUMNS = (
    "MolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "LabuteASA",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "FractionCSP3",
    "HeavyAtomCount",
    "BertzCT",
    "TotalSubstituentCount",
    "OrthoSubstituentCount",
    "MetaSubstituentCount",
    "ParaSubstituentCount",
    "HalogenAtomCount",
    "HeteroatomsOutsideCOOH",
    "AdditionalAromaticRingCount",
    "SubstituentSP3CarbonCount",
    "GasteigerCharge_AcidicH",
    "GasteigerCharge_HydroxylO",
    "GasteigerCharge_CarbonylO",
    "GasteigerCharge_CarboxylC",
    "GasteigerCharge_IpsoC",
    "GasteigerCharge_OrthoCMean",
    "GasteigerCharge_MetaCMean",
    "GasteigerCharge_ParaC",
    "EState_HydroxylO",
    "EState_CarbonylO",
)
_ERROR_POLICIES = frozenset({"continue", "raise"})
_SOURCE_ATOM_PROPERTY = "_benzoic_descriptor_source_index"


def _same_file(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    return first.exists() and second.exists() and first.samefile(second)


def _validate_iterations(iterations: int) -> None:
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("gasteiger_iterations must be a positive integer.")


@dataclass(frozen=True)
class DescriptorExtractionConfig:
    """CSV抽出の入出力条件。元ファイルのGaussian専用設定は不要。"""

    input_path: Path | str
    output_path: Path | str
    invalid_output_path: Path | str | None = None
    name_column: str = "name"
    smiles_column: str = "smiles"
    input_encoding: str = "utf-8-sig"
    output_encoding: str = "utf-8-sig"
    on_error: str = "continue"
    gasteiger_iterations: int = 12

    def __post_init__(self) -> None:
        if self.on_error not in _ERROR_POLICIES:
            raise ValueError("on_error must be 'continue' or 'raise'.")
        _validate_iterations(self.gasteiger_iterations)
        codecs.lookup(self.input_encoding)
        codecs.lookup(self.output_encoding)
        if not self.name_column.strip() or not self.smiles_column.strip():
            raise ValueError("name_column and smiles_column must not be empty.")
        if self.name_column == self.smiles_column:
            raise ValueError("name_column and smiles_column must be different.")
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_path", Path(self.output_path))
        invalid_output_path = self.invalid_output_path
        if invalid_output_path is None:
            suffix = self.output_path.suffix or ".csv"
            invalid_output_path = self.output_path.with_name(
                f"{self.output_path.stem}_invalid{suffix}"
            )
        object.__setattr__(self, "invalid_output_path", Path(invalid_output_path))
        if _same_file(self.input_path, self.output_path):
            raise ValueError("Input and output files must be different.")
        if _same_file(self.input_path, self.invalid_output_path):
            raise ValueError("Input and invalid-output files must be different.")
        if _same_file(self.output_path, self.invalid_output_path):
            raise ValueError("Output and invalid-output files must be different.")


@dataclass
class DescriptorExtractionResult:
    config: DescriptorExtractionConfig
    dataframe: Any
    invalid_dataframe: Any
    rows_read: int
    rows_written: int
    error_counts: dict[str, int]

    @property
    def rows_excluded(self) -> int:
        return self.rows_read - self.rows_written


class DescriptorExtractionError(RuntimeError):
    """入力行の名前、SMILES、骨格、局所原子、計算に問題がある場合の例外。"""


@dataclass(frozen=True)
class _BenzoicAcidSite:
    carboxyl_carbon: int
    carbonyl_oxygen: int
    hydroxyl_oxygen: int
    ipso_carbon: int
    ring_atoms: tuple[int, ...]
    ortho_carbons: tuple[int, ...]
    meta_carbons: tuple[int, ...]
    para_carbon: int

    @property
    def carboxyl_atoms(self) -> frozenset[int]:
        return frozenset(
            (self.carboxyl_carbon, self.carbonyl_oxygen, self.hydroxyl_oxygen)
        )


def _import_dependencies() -> dict[str, Any]:
    try:
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, Lipinski
        from rdkit.Chem import rdMolDescriptors, rdPartialCharges
        from rdkit.Chem.EState.EState import EStateIndices
    except ImportError as exc:
        raise RuntimeError(
            "Descriptor extraction requires RDKit and pandas. "
            "Install them with: python -m pip install rdkit pandas"
        ) from exc
    return {
        "pd": pd,
        "Chem": Chem,
        "Crippen": Crippen,
        "Descriptors": Descriptors,
        "GraphDescriptors": GraphDescriptors,
        "Lipinski": Lipinski,
        "rdMolDescriptors": rdMolDescriptors,
        "rdPartialCharges": rdPartialCharges,
        "EStateIndices": EStateIndices,
    }


def _parse_smiles(smiles: str, dependencies: Mapping[str, Any]) -> Any:
    if not isinstance(smiles, str) or not smiles.strip():
        raise DescriptorExtractionError("SMILES is empty or is not a string.")
    Chem = dependencies["Chem"]
    parameters = Chem.SmilesParserParams()
    parameters.parseName = False
    parameters.allowCXSMILES = False
    try:
        mol = Chem.MolFromSmiles(smiles.strip(), parameters)
    except Exception as exc:
        raise DescriptorExtractionError("SMILES could not be parsed by RDKit.") from exc
    if mol is None or mol.GetNumAtoms() == 0:
        raise DescriptorExtractionError("SMILES could not be parsed by RDKit.")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise DescriptorExtractionError(
            "SMILES must describe one connected molecule; salts/fragments are unsupported."
        )
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise DescriptorExtractionError("Dummy/query atoms are unsupported.")
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        raise DescriptorExtractionError("Radical molecules are unsupported.")
    mol = Chem.RemoveHs(mol)
    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        raise DescriptorExtractionError(
            "Non-removable explicit H atoms (e.g. isotopic H) are unsupported."
        )
    return mol


def _mapped_matches(
    mol: Any, smarts: str, dependencies: Mapping[str, Any]
) -> tuple[dict[int, int], ...]:
    pattern = dependencies["Chem"].MolFromSmarts(smarts)
    if pattern is None:
        raise RuntimeError(f"Internal SMARTS is invalid: {smarts}")
    map_positions = {
        atom.GetAtomMapNum(): atom.GetIdx() for atom in pattern.GetAtoms()
    }
    if set(map_positions) != {1, 2, 3}:
        raise RuntimeError("Internal carboxyl SMARTS has invalid atom maps.")
    return tuple(
        {label: int(match[position]) for label, position in map_positions.items()}
        for match in mol.GetSubstructMatches(pattern, uniquify=True)
    )


def _aromatic_acid_attachment(mol: Any, acid: Mapping[int, int]) -> int | None:
    """COOHの唯一の環側隣接原子を返す。脂肪族酸ならNone。"""
    neighbors = [
        atom.GetIdx()
        for atom in mol.GetAtomWithIdx(acid[1]).GetNeighbors()
        if atom.GetIdx() not in {acid[2], acid[3]}
    ]
    if len(neighbors) != 1:
        return None
    ipso = neighbors[0]
    ipso_atom = mol.GetAtomWithIdx(ipso)
    if ipso_atom.GetAtomicNum() != 6 or not ipso_atom.GetIsAromatic():
        return None
    return ipso


def _map_benzoic_acid_site(
    mol: Any, acid: Mapping[int, int], ipso: int
) -> _BenzoicAcidSite:
    """酸SMARTSのmap番号と基準環内距離から局所原子を対応付ける。"""
    carboxyl_carbon, carbonyl_oxygen, hydroxyl_oxygen = (
        acid[1], acid[2], acid[3]
    )

    def is_aromatic_ring(ring: Sequence[int]) -> bool:
        return all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring) and all(
            mol.GetBondBetweenAtoms(index, ring[(position + 1) % len(ring)]).GetIsAromatic()
            for position, index in enumerate(ring)
        )

    aromatic_rings = [
        tuple(int(index) for index in ring)
        for ring in mol.GetRingInfo().AtomRings()
        if is_aromatic_ring(ring)
    ]
    candidates = [
        ring
        for ring in aromatic_rings
        if ipso in ring and len(ring) == 6
        and all(mol.GetAtomWithIdx(index).GetAtomicNum() == 6 for index in ring)
    ]
    if len(candidates) != 1:
        raise DescriptorExtractionError(
            "Benzoic acid scaffold mismatch/ambiguity: exactly one six-membered "
            "all-carbon aromatic ring must contain the COOH attachment."
        )
    ring = candidates[0]
    ring_set = frozenset(ring)
    # 環外の橋や側鎖を経由する近道を使わず、基準環内だけで距離を求める。
    adjacency = {
        index: tuple(
            atom.GetIdx()
            for atom in mol.GetAtomWithIdx(index).GetNeighbors()
            if atom.GetIdx() in ring_set
        )
        for index in ring
    }
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise DescriptorExtractionError("The reference benzene cycle is ambiguous.")
    if any(
        not mol.GetBondBetweenAtoms(index, neighbor).GetIsAromatic()
        for index, neighbors in adjacency.items() for neighbor in neighbors
    ):
        raise DescriptorExtractionError("The reference ring bonds must be aromatic.")
    distances = {ipso: 0}
    queue = deque([ipso])
    while queue:
        index = queue.popleft()
        for neighbor in adjacency[index]:
            if neighbor not in distances:
                distances[neighbor] = distances[index] + 1
                queue.append(neighbor)
    if Counter(distances.values()) != Counter({0: 1, 1: 2, 2: 2, 3: 1}):
        raise DescriptorExtractionError("Ortho/meta/para atom mapping is ambiguous.")
    return _BenzoicAcidSite(
        carboxyl_carbon=carboxyl_carbon,
        carbonyl_oxygen=carbonyl_oxygen,
        hydroxyl_oxygen=hydroxyl_oxygen,
        ipso_carbon=ipso,
        ring_atoms=tuple(sorted(ring)),
        ortho_carbons=tuple(sorted(i for i, distance in distances.items() if distance == 1)),
        meta_carbons=tuple(sorted(i for i, distance in distances.items() if distance == 2)),
        para_carbon=next(i for i, distance in distances.items() if distance == 3),
    )


def _find_benzoic_acid_sites(
    mol: Any, dependencies: Mapping[str, Any]
) -> tuple[_BenzoicAcidSite, ...]:
    """芳香環直結の中性COOHをすべて安全にマッピングする。"""
    anionic_sites = _mapped_matches(mol, CARBOXYLATE_SMARTS, dependencies)
    if any(_aromatic_acid_attachment(mol, acid) is not None for acid in anionic_sites):
        raise DescriptorExtractionError(
            "Aromatic carboxylates are unsupported: every aromatic acid site "
            "must be a neutral COOH with an acidic H."
        )
    sites: list[_BenzoicAcidSite] = []
    seen_carbons: set[int] = set()
    for acid in _mapped_matches(mol, ACID_SMARTS, dependencies):
        ipso = _aromatic_acid_attachment(mol, acid)
        if ipso is None:
            continue
        if acid[1] in seen_carbons:
            raise DescriptorExtractionError("The neutral COOH atom mapping is ambiguous.")
        seen_carbons.add(acid[1])
        try:
            sites.append(_map_benzoic_acid_site(mol, acid, ipso))
        except DescriptorExtractionError as exc:
            raise DescriptorExtractionError(
                f"Aromatic COOH at atom {acid[1]} cannot be mapped: {exc}"
            ) from exc
    if not sites:
        raise DescriptorExtractionError(
            "At least one neutral COOH directly bonded to a six-membered "
            "all-carbon aromatic ring is required; aliphatic acids are ignored."
        )
    return tuple(sites)


def _find_benzoic_acid_site(
    mol: Any, dependencies: Mapping[str, Any]
) -> _BenzoicAcidSite:
    """単一酸点を必要とする既存コード用。複数酸点を任意に選ばない。"""
    sites = _find_benzoic_acid_sites(mol, dependencies)
    if len(sites) != 1:
        raise DescriptorExtractionError(
            f"Found {len(sites)} aromatic COOH sites; use the multi-site API."
        )
    return sites[0]


def calculate_rdkit_descriptors(
    mol: Any, dependencies: Mapping[str, Any]
) -> dict[str, float | int]:
    Descriptors = dependencies["Descriptors"]
    Crippen = dependencies["Crippen"]
    Lipinski = dependencies["Lipinski"]
    rdMolDescriptors = dependencies["rdMolDescriptors"]
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "MolLogP": float(Crippen.MolLogP(mol)),
        "MolMR": float(Crippen.MolMR(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "LabuteASA": float(rdMolDescriptors.CalcLabuteASA(mol)),
        "NumHAcceptors": int(Lipinski.NumHAcceptors(mol)),
        "NumHDonors": int(Lipinski.NumHDonors(mol)),
        "NumHeteroatoms": int(Lipinski.NumHeteroatoms(mol)),
        "NumRotatableBonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol, True)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "HeavyAtomCount": int(mol.GetNumHeavyAtoms()),
        "BertzCT": float(dependencies["GraphDescriptors"].BertzCT(mol)),
    }


def calculate_structural_descriptors(
    mol: Any, site: _BenzoicAcidSite, dependencies: Mapping[str, Any]
) -> dict[str, int]:
    ring_set = frozenset(site.ring_atoms)
    core_atoms = ring_set | site.carboxyl_atoms

    def count_substituted_positions(indices: Sequence[int]) -> int:
        # 縮合共有炭素も環外の原子に接続するので1置換位置として数える。
        # 同じ基が複数位置に接続する場合も各位置を1回ずつ数える。
        return sum(
            any(
                neighbor.GetAtomicNum() > 1
                and neighbor.GetIdx() not in ring_set
                and neighbor.GetIdx() != site.carboxyl_carbon
                for neighbor in mol.GetAtomWithIdx(index).GetNeighbors()
            )
            for index in indices
        )

    ortho = count_substituted_positions(site.ortho_carbons)
    meta = count_substituted_positions(site.meta_carbons)
    para = count_substituted_positions((site.para_carbon,))
    return {
        "TotalSubstituentCount": ortho + meta + para,
        "OrthoSubstituentCount": ortho,
        "MetaSubstituentCount": meta,
        "ParaSubstituentCount": para,
        "HalogenAtomCount": sum(
            atom.GetAtomicNum() in {9, 17, 35, 53} for atom in mol.GetAtoms()
        ),
        "HeteroatomsOutsideCOOH": sum(
            atom.GetAtomicNum() not in {1, 6}
            and atom.GetIdx() not in site.carboxyl_atoms
            for atom in mol.GetAtoms()
        ),
        "AdditionalAromaticRingCount": int(
            dependencies["rdMolDescriptors"].CalcNumAromaticRings(mol)
        ) - 1,
        "SubstituentSP3CarbonCount": sum(
            atom.GetAtomicNum() == 6
            and atom.GetHybridization() == dependencies["Chem"].HybridizationType.SP3
            and atom.GetIdx() not in core_atoms
            for atom in mol.GetAtoms()
        ),
    }


def _finite_number(value: Any, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DescriptorExtractionError(f"Invalid numeric value for {description}.") from exc
    if not math.isfinite(number):
        raise DescriptorExtractionError(f"Non-finite numeric value for {description}.")
    return number


def _calculate_local_descriptors_for_sites(
    mol: Any,
    sites: Sequence[_BenzoicAcidSite],
    dependencies: Mapping[str, Any],
    *,
    gasteiger_iterations: int = 12,
) -> tuple[dict[str, float], ...]:
    """全分子の電荷/EStateを1回計算し、各酸点の局所値を返す。"""
    Chem = dependencies["Chem"]
    tagged_mol = Chem.Mol(mol)
    for atom in tagged_mol.GetAtoms():
        atom.SetIntProp(_SOURCE_ATOM_PROPERTY, atom.GetIdx())
    charged_mol = Chem.AddHs(tagged_mol)
    index_map: dict[int, int] = {}
    for atom in charged_mol.GetAtoms():
        if atom.HasProp(_SOURCE_ATOM_PROPERTY):
            original_index = atom.GetIntProp(_SOURCE_ATOM_PROPERTY)
            if original_index in index_map:
                raise DescriptorExtractionError("Duplicate heavy-atom mapping after AddHs.")
            index_map[original_index] = atom.GetIdx()
    if set(index_map) != set(range(mol.GetNumAtoms())):
        raise DescriptorExtractionError("Heavy-atom mapping was lost after AddHs.")
    try:
        dependencies["rdPartialCharges"].ComputeGasteigerCharges(
            charged_mol,
            nIter=gasteiger_iterations,
            throwOnParamFailure=True,
        )
    except Exception as exc:
        raise DescriptorExtractionError(
            f"Gasteiger charge calculation failed (unsupported parameters or molecule): {exc}"
        ) from exc
    charges: dict[int, float] = {}
    for atom in charged_mol.GetAtoms():
        if not atom.HasProp("_GasteigerCharge"):
            raise DescriptorExtractionError(
                f"Gasteiger charge is missing for atom {atom.GetIdx()}."
            )
        charges[atom.GetIdx()] = _finite_number(
            atom.GetProp("_GasteigerCharge"), f"Gasteiger charge of atom {atom.GetIdx()}"
        )

    def charge(index: int) -> float:
        return charges[index_map[index]]

    estate = dependencies["EStateIndices"](mol)
    if len(estate) != mol.GetNumAtoms():
        raise DescriptorExtractionError("EState atom count does not match the heavy-atom molecule.")
    results: list[dict[str, float]] = []
    for site in sites:
        acid_hydrogens = [
            atom.GetIdx()
            for atom in charged_mol.GetAtomWithIdx(index_map[site.hydroxyl_oxygen]).GetNeighbors()
            if atom.GetAtomicNum() == 1
        ]
        if len(acid_hydrogens) != 1:
            raise DescriptorExtractionError(
                f"Exactly one acidic H must be attached to the OH oxygen "
                f"at atom {site.hydroxyl_oxygen}."
            )
        results.append({
            "GasteigerCharge_AcidicH": charges[acid_hydrogens[0]],
            "GasteigerCharge_HydroxylO": charge(site.hydroxyl_oxygen),
            "GasteigerCharge_CarbonylO": charge(site.carbonyl_oxygen),
            "GasteigerCharge_CarboxylC": charge(site.carboxyl_carbon),
            "GasteigerCharge_IpsoC": charge(site.ipso_carbon),
            "GasteigerCharge_OrthoCMean": sum(charge(i) for i in site.ortho_carbons) / 2.0,
            "GasteigerCharge_MetaCMean": sum(charge(i) for i in site.meta_carbons) / 2.0,
            "GasteigerCharge_ParaC": charge(site.para_carbon),
            "EState_HydroxylO": _finite_number(estate[site.hydroxyl_oxygen], "OH oxygen EState"),
            "EState_CarbonylO": _finite_number(estate[site.carbonyl_oxygen], "C=O oxygen EState"),
        })
    return tuple(results)


def calculate_local_descriptors(
    mol: Any,
    site: _BenzoicAcidSite,
    dependencies: Mapping[str, Any],
    *,
    gasteiger_iterations: int = 12,
) -> dict[str, float]:
    """単一酸点の局所記述子を返す既存API。"""
    return _calculate_local_descriptors_for_sites(
        mol, (site,), dependencies, gasteiger_iterations=gasteiger_iterations
    )[0]


def _mean_site_descriptors(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, float | int]:
    """酸点ごとの値を等重み平均。同じ整数値なら元の整数型を保つ。"""
    if not rows:
        raise DescriptorExtractionError("No aromatic COOH sites to average.")
    columns = tuple(rows[0])
    if any(tuple(row) != columns for row in rows):
        raise DescriptorExtractionError("Descriptor columns differ between acid sites.")
    result: dict[str, float | int] = {}
    for column in columns:
        values = [row[column] for row in rows]
        if all(isinstance(value, int) and value == values[0] for value in values):
            result[column] = values[0]
        else:
            result[column] = math.fsum(values) / len(values)
    return result


def calculate_descriptors(
    smiles: str,
    dependencies: Mapping[str, Any] | None = None,
    *,
    gasteiger_iterations: int = 12,
) -> dict[str, float | int]:
    """1分子の30特徴量を固定順序で返す。不適格分子では独自例外を送出。"""
    _validate_iterations(gasteiger_iterations)
    if dependencies is None:
        dependencies = _import_dependencies()
    try:
        mol = _parse_smiles(smiles, dependencies)
        sites = _find_benzoic_acid_sites(mol, dependencies)
        result: dict[str, float | int] = {}
        result.update(calculate_rdkit_descriptors(mol, dependencies))
        structural_rows = [
            calculate_structural_descriptors(mol, site, dependencies) for site in sites
        ]
        # ハロゲン数は全分子の値なので、酸点平均の対象から外す。
        result["HalogenAtomCount"] = structural_rows[0]["HalogenAtomCount"]
        result.update(_mean_site_descriptors([
            {column: value for column, value in row.items() if column != "HalogenAtomCount"}
            for row in structural_rows
        ]))
        result.update(_mean_site_descriptors(_calculate_local_descriptors_for_sites(
            mol, sites, dependencies, gasteiger_iterations=gasteiger_iterations
        )))
        expected, actual = set(DESCRIPTOR_COLUMNS), set(result)
        if actual != expected:
            raise DescriptorExtractionError(
                f"Descriptor columns are inconsistent; missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}."
            )
        for column, value in result.items():
            _finite_number(value, column)
        return {column: result[column] for column in DESCRIPTOR_COLUMNS}
    except DescriptorExtractionError:
        raise
    except Exception as exc:
        raise DescriptorExtractionError(
            f"Descriptor calculation failed: {type(exc).__name__}: {exc}"
        ) from exc


def _extract_row(
    row: Any,
    config: DescriptorExtractionConfig,
    dependencies: Mapping[str, Any],
) -> dict[str, float | int]:
    name = str(row[config.name_column]).strip()
    smiles = str(row[config.smiles_column]).strip()
    if not name:
        raise DescriptorExtractionError("Name is empty.")
    if not smiles:
        raise DescriptorExtractionError("SMILES is empty.")
    return calculate_descriptors(
        smiles, dependencies, gasteiger_iterations=config.gasteiger_iterations
    )


def _read_input_csv(config: DescriptorExtractionConfig, pd: Any) -> Any:
    # pandasは重複ヘッダーを自動改名するため、先に元のヘッダーを確認する。
    with config.input_path.open("r", encoding=config.input_encoding, newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next((row for row in reader if row), None)
        if header is None:
            raise ValueError("Input CSV is empty; a header is required.")
        if any(not column.strip() for column in header):
            raise ValueError("Input CSV must not contain empty column names.")
        if len(header) != len(set(header)):
            raise ValueError("Input data must not contain duplicate column names.")
        # index_col=Falseだけでは余分な列を警告付きで切り捨てる場合がある。
        # 列数不一致は行のデータを失わないよう、出力前にCSV全体のエラーとする。
        for record in reader:
            if record and len(record) != len(header):
                raise ValueError(
                    f"Malformed CSV record ending at line {reader.line_num}: "
                    f"expected {len(header)} columns, found {len(record)}."
                )
    dataframe = pd.read_csv(
        config.input_path,
        encoding=config.input_encoding,
        dtype=str,
        keep_default_na=False,
        index_col=False,
    )
    if list(dataframe.columns) != header:
        raise ValueError("Input CSV column names could not be preserved exactly.")
    missing = [
        column for column in (config.name_column, config.smiles_column)
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    reserved_columns = set(DESCRIPTOR_COLUMNS) | {
        "source_index", "error_type", "error_message"
    }
    conflicts = sorted(reserved_columns.intersection(dataframe.columns))
    if conflicts:
        raise ValueError("Input data already contains output columns: " + ", ".join(conflicts))
    return dataframe


def extract_descriptors(
    config: DescriptorExtractionConfig | Mapping[str, Any],
) -> DescriptorExtractionResult:
    """CSVから記述子を抽出し、成功行と失敗行をそれぞれ保存する。"""
    if not isinstance(config, DescriptorExtractionConfig):
        config = DescriptorExtractionConfig(**dict(config))
    dependencies = _import_dependencies()
    pd = dependencies["pd"]
    dataframe = _read_input_csv(config, pd)
    valid_rows: list[Any] = []
    feature_rows: list[dict[str, float | int]] = []
    invalid_rows: list[Any] = []
    error_counts: dict[str, int] = {}
    for source_index, row in dataframe.iterrows():
        try:
            descriptors = _extract_row(row, config, dependencies)
        except Exception as exc:
            if config.on_error == "raise":
                raise DescriptorExtractionError(
                    f"Row {source_index} failed: {type(exc).__name__}: {exc}"
                ) from exc
            invalid = row.copy()
            invalid["source_index"] = source_index
            invalid["error_type"] = type(exc).__name__
            invalid["error_message"] = str(exc)
            invalid_rows.append(invalid)
            error_counts[type(exc).__name__] = error_counts.get(type(exc).__name__, 0) + 1
            continue
        valid = row.copy()
        valid["source_index"] = source_index
        valid_rows.append(valid)
        feature_rows.append(descriptors)

    source_columns = list(dataframe.columns) + ["source_index"]
    valid_frame = pd.DataFrame(valid_rows, columns=source_columns).reset_index(drop=True)
    feature_frame = pd.DataFrame(feature_rows, columns=DESCRIPTOR_COLUMNS)
    output = pd.concat([valid_frame, feature_frame], axis=1)
    invalid_columns = list(dataframe.columns) + [
        "source_index", "error_type", "error_message"
    ]
    invalid_output = pd.DataFrame(invalid_rows, columns=invalid_columns).reset_index(drop=True)

    invalid_output_path = config.invalid_output_path
    if invalid_output_path is None:
        raise RuntimeError("invalid_output_path was not initialized.")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.output_path, index=False, encoding=config.output_encoding)
    invalid_output.to_csv(invalid_output_path, index=False, encoding=config.output_encoding)
    return DescriptorExtractionResult(
        config=config,
        dataframe=output,
        invalid_dataframe=invalid_output,
        rows_read=len(dataframe),
        rows_written=len(output),
        error_counts=error_counts,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="安息香酸誘導体のSMILESから30特徴量を抽出")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--invalid-output", type=Path, default=None)
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--input-encoding", default="utf-8-sig")
    parser.add_argument("--output-encoding", default="utf-8-sig")
    parser.add_argument("--on-error", choices=sorted(_ERROR_POLICIES), default="continue")
    parser.add_argument("--gasteiger-iterations", type=int, default=12)
    args = parser.parse_args(argv)
    config = DescriptorExtractionConfig(
        input_path=args.input_csv,
        output_path=args.output_csv,
        invalid_output_path=args.invalid_output,
        name_column=args.name_column,
        smiles_column=args.smiles_column,
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        on_error=args.on_error,
        gasteiger_iterations=args.gasteiger_iterations,
    )
    result = extract_descriptors(config)
    print(f"Rows read: {result.rows_read}; valid: {result.rows_written}; invalid: {result.rows_excluded}")
    print(f"Valid CSV: {config.output_path}")
    print(f"Invalid CSV: {config.invalid_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
