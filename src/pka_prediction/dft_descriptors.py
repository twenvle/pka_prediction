#!/usr/bin/env python3
"""安息香酸誘導体のGaussian log/fchkからDFT特徴量30個を取得する。

既存descriptors.pyおよびSMILES用ファイルは変更せずに作成した独立スクリプト。
Config/Result dataclass、独自例外、依存関係の遅延import、1分子計算、
CSV行単位処理、valid/invalid CSV、on_error方針を引き継ぐ。
依存関係は numpy, pandas, rdkit。Gaussianの計算そのものは実行しない。

使用例:
    python benzoic_acid_dft_descriptors.py input.csv valid.csv --log-dir logs --fchk-dir fchk
    python benzoic_acid_dft_descriptors.py input.csv valid.csv --cubegen /path/to/cubegen
    python benzoic_acid_dft_descriptors.py input.csv valid.csv --missing-policy raise

CSV必須列: name,smiles。既定のファイル名はsub_{name}.log/sub_{name}.fchk。
任意のlog_path,fchk_path列で各行のファイルを明示できる。CSV内の相対パスは
CSVの保存フォルダーを基準とする。ディレクトリ設定の相対パスは実行場所基準。
任意列density_cube_path,esp_cube_pathで各行のcubeファイルを明示できる。
Config/CLIでは密度cubeとESP cubeのフォルダーをそれぞれ別に指定できる。

30特徴量の定義/単位:
  局所23特徴量(1-14,22-30)は芳香環直結の各中性COOHで求め、等重み平均する。
  全分子7特徴量(15-21)は平均せず、脂肪族COOHも含む全分子から求める。
  脂肪族COOHは局所平均の対象外だが、電子状態やSASA遮蔽等からは除去しない。
  1酸点でも値が不足した局所特徴量は欠損とし、残りだけの平均は返さない。
  1-8  NPA電荷(e): 酸性H,OH側O,C=O側O,COOH炭素,ipso,
       ortho平均,meta平均,para。COOH炭素電荷は基全体の和ではない。
  9-12 Wiberg指数: O-H,C-O(H),C=O,aryl-C(COOH)。NAO行列から取得。
 13-14 表面ESP(Hartree/e): rho=0.001 e/Bohr^3等値面の格子辺交点で
       ESPを線形補間し、酸性H/C=O酸素に最も近い核近傍(半径2 A)の
       面領域における最大/最小。点電荷や核位置のESPで代用しない。
 15-17 HOMO,LUMO,gap(eV):中性閉殻一重項のfchk軌道エネルギー。
 18    双極子(Debye): fchk Dipole Momentのノルムを換算。
 19    等方分極率(a.u.): packed tensor XX,XY,YY,XZ,YZ,ZZのtrace/3。
 20    MPI(Hartree/e): rho=0.001 e/Bohr^3等値面の全格子辺交点で線形補間した
       |ESP|の算術平均。格子による離散近似なので全分子で同じ等値面と解像度を使う。
 21    分子体積(A^3): 上記電子密度等値面内の格子点数*voxel体積。
 22    COOH局所SASA(A^2):最適化座標、RDKit vdW半径、probe1.4 Aを
       用いたFibonacci球面サンプリング。C,O,O,酸性Hの露出面積の和。
 23    COOH近傍buried volume(%): COOH炭素を中心とする半径3.5 A球内で、
       COOH4原子以外のvdW球に占有される割合。Halton体積サンプリング。
 24-27 結合長(A): O-H,C=O,C-O(H),aryl-C(COOH)。fchk座標から計算。
 28    ベンゼン-COOH面角(deg,0-90): ベンゼン6炭素の最小二乗平面と
       C,O,O面のなす角。ortho左右や原子順序によらないねじれ角の大きさ。
 29-30 O-H/C=O伸縮振動(cm^-1): 全3N-6調和モードのうち、正の周波数で
       正規化した結合軸伸縮投影が最大のモードを割り当てる。
       結合独立の局所モード周波数ではなく、混成した正常モードの割当。
       複数酸点が同じモードに割り当てられる場合もある。各酸点の割当値を平均する。
       frequency_scale既定1.0。顕著な虚振動は既定で行エラー。
       fchkに残った過去のHessianを使わないよう、最終logの周波数と照合する。

電子密度/ESP cubeが両方ない場合、--cubegen指定またはPATH上のGaussian
cubegenでfchkからFDensity=SCFとPotential=SCFを同じ格子に生成する。
生成名は{logファイルのstem}_density.cube/{logファイルのstem}_esp.cube。
cubeは一時フォルダーに作り処理後に削除。fchk/log/既存cubeを変更しない。
体積、表面ESP、SASA、buried値には数値サンプリング誤差があるため、
設定を全分子で統一し、必要に応じて解像度依存性を確認する。

missing_policy='nan'(既定): 不足する計算出力は欠損値とし、
descriptor_status=partial、missing_descriptors、descriptor_issuesに明記。
partial行もvalid CSVに含む。完全な30特徴量だけを採用するにはraiseを指定。
on_error='continue'は構造不適格・ファイル不一致等をinvalid CSVに分離し、
'raise'では最初の失敗で停止してCSVを書き込まない。ID先頭ゼロ/文字列NA保持。

酸点は6員の全炭素芳香環に直結する中性COOHすべて。ナフタレン等の縮合環も対応。
ortho/meta/paraは各基準環内だけの距離で定義し、縮合部を含む環外への接続位置を
置換位置として数える。酸点数、全局所原子番号、各置換位置数はprovenanceに記録。
芳香環直結カルボキシラート、不一致/曖昧な基準環、ラジカル、同位体標識H、
開殻は対象外。脂肪族COOHしかない分子は対象酸点がないためエラー。
SMILESは原子同定だけに使い、SMILES由来の電子記述子をDFT列に混ぜない。
元素だけで原子順序を判断せず、座標由来の結合関係/H数をSMILESと照合する。
logは正常終了した最後のジョブだけを使い、以前のジョブのNBO出力を流用しない。
原子列、構造距離、利用可能なSCFエネルギーでlog/fchkの一致を検証する。

仕様参考(Gaussian作成マニュアルの公開ミラー、RDKit、cclibの実装):
https://theochem.mercer.edu/chm295/g09ur/u_cubegen.htm
https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html
https://github.com/cclib/cclib/blob/master/cclib/parser/fchkparser.py
"""
from __future__ import annotations

import argparse
import codecs
import csv
import json
import math
import re
import shutil
import tempfile
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["DescriptorExtractionConfig", "DescriptorExtractionResult",
           "DescriptorExtractionError", "DESCRIPTOR_COLUMNS",
           "calculate_descriptors", "extract_descriptors"]

DESCRIPTOR_COLUMNS = (
    "npa_acidic_h", "npa_hydroxyl_o", "npa_carbonyl_o", "npa_carboxyl_c",
    "npa_ipso_c", "npa_ortho_c_mean", "npa_meta_c_mean", "npa_para_c",
    "wiberg_oh", "wiberg_co_single", "wiberg_co_double", "wiberg_aryl_cooh",
    "esp_max_acidic_h_hartree_per_e", "esp_min_carbonyl_o_hartree_per_e",
    "homo_ev", "lumo_ev", "homo_lumo_gap_ev", "dipole_moment_debye",
    "isotropic_polarizability_au", "mpi_hartree_per_e",
    "molecular_volume_angstrom3", "cooh_local_sasa_angstrom2",
    "cooh_buried_volume_percent", "oh_bond_length_angstrom",
    "co_double_bond_length_angstrom", "co_single_bond_length_angstrom",
    "aryl_cooh_bond_length_angstrom", "benzene_cooh_plane_angle_deg",
    "oh_stretch_frequency_cm1", "co_double_stretch_frequency_cm1",
)
METADATA_COLUMNS = ("descriptor_status", "missing_descriptors", "descriptor_issues",
                    "descriptor_provenance")
HARTREE_TO_EV = 27.211386245988
DIPOLE_AU_TO_DEBYE = 2.541746473
ACID_SMARTS = "[C;X3;+0:1](=[O;X1;+0:2])-[O;X2;H1;+0:3]"
CARBOXYLATE_SMARTS = "[C;X3;+0:1](=[O;X1;+0:2])-[O;X1;-1:3]"
_ERROR_POLICIES = frozenset({"continue", "raise"})


class DescriptorExtractionError(RuntimeError):
    """Input structure, file pairing, or requested DFT features are invalid."""


class _UnsafeOutputPath(ValueError):
    pass


def _same_file(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve() or (
        first.exists() and second.exists() and first.samefile(second)
    )


def _finite_scalar(value: Any, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid numeric value: {description}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {description}")
    return result


@dataclass(frozen=True)
class DescriptorExtractionConfig:
    input_path: Path | str
    output_path: Path | str
    logdata_path: Path | str = "."
    fchkdata_path: Path | str | None = None
    density_cube_path: Path | str | None = None
    esp_cube_path: Path | str | None = None
    invalid_output_path: Path | str | None = None
    name_column: str = "name"
    smiles_column: str = "smiles"
    log_column: str = "log_path"
    fchk_column: str = "fchk_path"
    density_cube_column: str = "density_cube_path"
    esp_cube_column: str = "esp_cube_path"
    file_stem_template: str = "sub_{name}"
    input_encoding: str = "utf-8-sig"
    output_encoding: str = "utf-8-sig"
    log_encoding: str = "utf-8"
    on_error: str = "continue"
    missing_policy: str = "nan"
    cubegen_path: Path | str | None = None
    cube_npts: int = 100
    cube_timeout_seconds: int = 600
    density_isovalue: float = 0.001
    esp_local_radius_angstrom: float = 2.0
    sasa_probe_radius_angstrom: float = 1.4
    sasa_points_per_atom: int = 2000
    buried_radius_angstrom: float = 3.5
    buried_samples: int = 32768
    geometry_tolerance_angstrom: float = 0.02
    connectivity_factor: float = 1.25
    frequency_scale: float = 1.0
    min_stretch_projection: float = 0.05
    require_minimum: bool = True
    imaginary_frequency_tolerance_cm1: float = 20.0

    def __post_init__(self) -> None:
        if self.on_error not in _ERROR_POLICIES or self.missing_policy not in {"nan", "raise"}:
            raise ValueError("on_error must be continue/raise; missing_policy must be nan/raise.")
        for encoding in (self.input_encoding, self.output_encoding, self.log_encoding):
            codecs.lookup(encoding)
        columns = [self.name_column, self.smiles_column, self.log_column, self.fchk_column,
                   self.density_cube_column, self.esp_cube_column]
        if any(not isinstance(c, str) or not c.strip() for c in columns) or len(set(columns)) != len(columns):
            raise ValueError("Input column settings must be distinct nonempty strings.")
        for attribute in ("input_path", "output_path", "logdata_path"):
            object.__setattr__(self, attribute, Path(getattr(self, attribute)))
        object.__setattr__(self, "fchkdata_path", Path(self.fchkdata_path or self.logdata_path))
        for attribute in ("density_cube_path", "esp_cube_path"):
            value = getattr(self, attribute)
            if value is not None:
                object.__setattr__(self, attribute, Path(value))
        invalid = self.invalid_output_path or self.output_path.with_name(
            f"{self.output_path.stem}_invalid{self.output_path.suffix or '.csv'}"
        )
        object.__setattr__(self, "invalid_output_path", Path(invalid))
        if self.cubegen_path is not None:
            object.__setattr__(self, "cubegen_path", str(self.cubegen_path))
        for first, second in ((self.input_path, self.output_path),
                              (self.input_path, self.invalid_output_path),
                              (self.output_path, self.invalid_output_path)):
            if _same_file(first, second):
                raise ValueError("Input, valid output and invalid output must be different files.")
        for output in (self.output_path, self.invalid_output_path):
            if _same_file(output, Path(__file__)):
                raise ValueError("Output cannot overwrite this Python module.")
        if "{name}" not in self.file_stem_template:
            raise ValueError("file_stem_template must contain {name}.")
        try:
            test_stem = self.file_stem_template.format(name="test")
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError("Invalid file_stem_template.") from exc
        if not test_stem or any(c in test_stem for c in '*?[]/\\:') or test_stem in {'.', '..'}:
            raise ValueError("file_stem_template must produce a plain filename stem.")
        for attribute, minimum in (("cube_npts", 10), ("cube_timeout_seconds", 1),
                                   ("sasa_points_per_atom", 100), ("buried_samples", 1000)):
            value = getattr(self, attribute)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{attribute} must be an integer >= {minimum}.")
        if self.cube_npts > 125:
            raise ValueError("cube_npts must be <=125 (scalar-grid memory limit).")
        for attribute in ("density_isovalue", "esp_local_radius_angstrom",
                          "buried_radius_angstrom", "geometry_tolerance_angstrom",
                          "connectivity_factor", "frequency_scale", "min_stretch_projection"):
            if _finite_scalar(getattr(self, attribute), attribute) <= 0:
                raise ValueError(f"{attribute} must be positive.")
        for attribute in ("sasa_probe_radius_angstrom", "imaginary_frequency_tolerance_cm1"):
            if _finite_scalar(getattr(self, attribute), attribute) < 0:
                raise ValueError(f"{attribute} must be nonnegative.")
        if not isinstance(self.require_minimum, bool):
            raise ValueError("require_minimum must be a boolean.")


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

    @property
    def rows_partial(self) -> int:
        return int((self.dataframe["descriptor_status"] == "partial").sum())


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
        return frozenset((self.carboxyl_carbon, self.carbonyl_oxygen, self.hydroxyl_oxygen))


@dataclass(frozen=True)
class _DftSite:
    carboxyl_carbon: int
    carbonyl_oxygen: int
    hydroxyl_oxygen: int
    acidic_hydrogen: int
    ipso_carbon: int
    ring_atoms: tuple[int, ...]
    ortho_carbons: tuple[int, ...]
    meta_carbons: tuple[int, ...]
    para_carbon: int
    substituted_ring_carbons: tuple[int, ...] = ()

    @property
    def carboxyl_atoms(self) -> frozenset[int]:
        return frozenset((self.carboxyl_carbon, self.carbonyl_oxygen,
                          self.hydroxyl_oxygen, self.acidic_hydrogen))


def _import_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError as exc:
        raise RuntimeError("Install numpy, pandas and RDKit: python -m pip install numpy pandas rdkit") from exc
    return {"np": np, "pd": pd, "Chem": Chem, "rdDetermineBonds": rdDetermineBonds}


def _default_config() -> DescriptorExtractionConfig:
    return DescriptorExtractionConfig("__unused_input__.csv", "__unused_valid__.csv")


def _prepare_geometry(smiles: str, fchk: Mapping[str, Any], dependencies: Mapping[str, Any],
                      config: DescriptorExtractionConfig | None = None) -> tuple[tuple[_DftSite, ...], Any]:
    """Match element/connectivity/H counts, never assume SMILES=fchk atom order."""
    config = config or _default_config()
    np, Chem = dependencies["np"], dependencies["Chem"]
    template = _parse_smiles(smiles, dependencies)
    anchors = _find_benzoic_acid_sites(template, dependencies)
    numbers = np.asarray(fchk["atomic_numbers"], dtype=int)
    coordinates = np.asarray(fchk["coordinates_angstrom"], dtype=float)
    if coordinates.shape != (len(numbers), 3) or not np.isfinite(coordinates).all():
        raise DescriptorExtractionError("Invalid fchk Cartesian coordinates.")
    if int(fchk["charge"]) != 0 or int(fchk["multiplicity"]) != 1:
        raise DescriptorExtractionError("The DFT extractor requires a neutral singlet acid.")
    if sum(atom.GetFormalCharge() for atom in template.GetAtoms()) != 0:
        raise DescriptorExtractionError("SMILES must describe the same neutral acid state.")
    explicit_template = Chem.AddHs(template)
    if sorted(numbers.tolist()) != sorted(atom.GetAtomicNum() for atom in explicit_template.GetAtoms()):
        raise DescriptorExtractionError("SMILES/fchk elemental composition or hydrogen count differs.")
    xyz_lines = [str(len(numbers)), "fchk geometry"] + [
        f"{Chem.GetPeriodicTable().GetElementSymbol(int(z))} {x:.10f} {y:.10f} {zcoord:.10f}"
        for z, (x, y, zcoord) in zip(numbers, coordinates)
    ]
    geometry = Chem.MolFromXYZBlock("\n".join(xyz_lines) + "\n")
    if geometry is None:
        raise DescriptorExtractionError("RDKit could not construct the fchk XYZ geometry.")
    try:
        dependencies["rdDetermineBonds"].DetermineConnectivity(
            geometry, useVdw=True, covFactor=config.connectivity_factor
        )
    except Exception as exc:
        raise DescriptorExtractionError(f"Could not determine geometric connectivity: {exc}") from exc
    heavy_indices = [i for i, z in enumerate(numbers) if z != 1]
    heavy_map = {original: new for new, original in enumerate(heavy_indices)}
    target = Chem.RWMol()
    h_neighbors: dict[int, tuple[int, ...]] = {}
    for original in heavy_indices:
        atom = geometry.GetAtomWithIdx(original)
        hydrogens = tuple(n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
        h_neighbors[original] = hydrogens
        new_atom = Chem.Atom(int(numbers[original]))
        new_atom.SetNoImplicit(True)
        new_atom.SetNumExplicitHs(len(hydrogens))
        target.AddAtom(new_atom)
    for hydrogen in (i for i, z in enumerate(numbers) if z == 1):
        neighbors = list(geometry.GetAtomWithIdx(hydrogen).GetNeighbors())
        if len(neighbors) != 1 or neighbors[0].GetAtomicNum() == 1:
            raise DescriptorExtractionError("A geometric H does not have one heavy-atom neighbor.")
    for bond in geometry.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in heavy_map and right in heavy_map:
            target.AddBond(heavy_map[left], heavy_map[right], Chem.BondType.SINGLE)
    target = target.GetMol()
    target.UpdatePropertyCache(strict=False)
    query = Chem.RWMol()
    for atom in template.GetAtoms():
        query.AddAtom(Chem.AtomFromSmarts(
            f"[#{atom.GetAtomicNum()};H{atom.GetTotalNumHs(includeNeighbors=True)};D{atom.GetDegree()}]"
        ))
    for bond in template.GetBonds():
        query.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
        query.ReplaceBond(query.GetNumBonds() - 1, Chem.BondFromSmarts("~"))
    query = query.GetMol()
    if target.GetNumBonds() != template.GetNumBonds():
        raise DescriptorExtractionError("SMILES/fchk heavy-atom connectivity differs.")
    matches = target.GetSubstructMatches(query, uniquify=False, maxMatches=4097)
    if not matches or len(matches) >= 4097:
        raise DescriptorExtractionError("SMILES/fchk atom mapping failed or exceeds the ambiguity limit.")
    site_sets: set[frozenset[_DftSite]] = set()
    for match in matches:
        def mapped(index: int) -> int:
            return heavy_indices[match[index]]
        mapped_sites: set[_DftSite] = set()
        for anchor in anchors:
            hydroxyl = mapped(anchor.hydroxyl_oxygen)
            if len(h_neighbors[hydroxyl]) != 1:
                raise DescriptorExtractionError("An aromatic COOH lacks one mapped acidic H.")
            ring_set = frozenset(anchor.ring_atoms)
            substituted = tuple(sorted(mapped(i) for i in anchor.ring_atoms if any(
                neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in ring_set
                and neighbor.GetIdx() != anchor.carboxyl_carbon
                for neighbor in template.GetAtomWithIdx(i).GetNeighbors()
            )))
            mapped_sites.add(_DftSite(
                mapped(anchor.carboxyl_carbon), mapped(anchor.carbonyl_oxygen), hydroxyl,
                h_neighbors[hydroxyl][0], mapped(anchor.ipso_carbon),
                tuple(sorted(mapped(i) for i in anchor.ring_atoms)),
                tuple(sorted(mapped(i) for i in anchor.ortho_carbons)),
                tuple(sorted(mapped(i) for i in anchor.meta_carbons)), mapped(anchor.para_carbon),
                substituted,
            ))
        if len(mapped_sites) != len(anchors) or len({s.carboxyl_carbon for s in mapped_sites}) != len(anchors):
            raise DescriptorExtractionError("The complete aromatic COOH mapping lost or duplicated an acid site.")
        site_sets.add(frozenset(mapped_sites))
    if len(site_sets) != 1:
        raise DescriptorExtractionError("The complete set of DFT acid/local atom mappings is not unique.")
    sites = tuple(sorted(next(iter(site_sets)), key=lambda site: site.carboxyl_carbon))
    return sites, coordinates


def _pairwise_distances(coordinates: Any, np: Any) -> Any:
    return np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)


def _validate_log_fchk(job_text: str, fchk: Mapping[str, Any], dependencies: Mapping[str, Any],
                       config: DescriptorExtractionConfig) -> Any:
    np = dependencies["np"]
    orientation = last_cartesian_orientation(job_text)
    if not np.array_equal(orientation["atomic_numbers"], fchk["atomic_numbers"]):
        raise DescriptorExtractionError("Log and fchk atom numbers/order differ.")
    log_coordinates = np.asarray(orientation["coordinates_angstrom"], dtype=float)
    difference = np.max(np.abs(_pairwise_distances(log_coordinates, np) -
                               _pairwise_distances(np.asarray(fchk["coordinates_angstrom"]), np)))
    if difference > config.geometry_tolerance_angstrom:
        raise DescriptorExtractionError("Log and fchk describe different geometries or atom orders.")
    states = re.findall(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", job_text)
    if states and tuple(map(int, states[-1])) != (int(fchk["charge"]), int(fchk["multiplicity"])):
        raise DescriptorExtractionError("Log and fchk charge/multiplicity differ.")
    energy, method = _scf_energy(job_text)
    if method.startswith(("U", "RO")):
        raise DescriptorExtractionError("Unrestricted/open-shell calculations are outside the closed-shell acid scope.")
    beta = fchk.get("beta_orbital_energies_hartree")
    alpha = fchk.get("alpha_orbital_energies_hartree")
    if beta is not None and alpha is not None and (
        np.asarray(beta).shape != np.asarray(alpha).shape or not np.allclose(beta, alpha, rtol=0, atol=1e-6)
    ):
        raise DescriptorExtractionError("A broken-symmetry or inconsistent alpha/beta checkpoint is unsupported.")
    checkpoint_energy = fchk.get("total_energy_hartree")
    if checkpoint_energy is None:
        checkpoint_energy = fchk.get("fields", {}).get("SCF Energy")
    if checkpoint_energy is not None and abs(energy - float(checkpoint_energy)) > 1e-5:
        raise DescriptorExtractionError("Log and fchk SCF energies differ; file pairing is unsafe.")
    return log_coordinates


def _scf_energy(text: str) -> tuple[float, str]:
    matches = re.findall(r"SCF Done:\s*E\(([^)]+)\)\s*=\s*([-+0-9.DEded]+)", text)
    if not matches:
        raise ValueError("No final SCF energy was found.")
    method, energy = matches[-1]
    return _finite_scalar(energy.replace("D", "E").replace("d", "e"), "SCF energy"), method.upper()


def _geometry_features(site: _DftSite, coordinates: Any, np: Any) -> dict[str, float]:
    def distance(left: int, right: int) -> float:
        return float(np.linalg.norm(coordinates[left] - coordinates[right]))
    ring_points = coordinates[list(site.ring_atoms)]
    _, singular_values, vectors = np.linalg.svd(ring_points - ring_points.mean(axis=0))
    if singular_values[1] < 1e-8:
        raise ValueError("The aromatic-ring plane is degenerate.")
    ring_normal = vectors[-1]
    acid_normal = np.cross(coordinates[site.carbonyl_oxygen] - coordinates[site.carboxyl_carbon],
                           coordinates[site.hydroxyl_oxygen] - coordinates[site.carboxyl_carbon])
    acid_norm = np.linalg.norm(acid_normal)
    if acid_norm < 1e-8:
        raise ValueError("The COOH plane is degenerate.")
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(ring_normal, acid_normal / acid_norm)), 0, 1))))
    return {
        "oh_bond_length_angstrom": distance(site.hydroxyl_oxygen, site.acidic_hydrogen),
        "co_double_bond_length_angstrom": distance(site.carboxyl_carbon, site.carbonyl_oxygen),
        "co_single_bond_length_angstrom": distance(site.carboxyl_carbon, site.hydroxyl_oxygen),
        "aryl_cooh_bond_length_angstrom": distance(site.ipso_carbon, site.carboxyl_carbon),
        "benzene_cooh_plane_angle_deg": angle,
    }


def _vdw_radii(numbers: Any, dependencies: Mapping[str, Any]) -> Any:
    radii = dependencies["np"].array([
        dependencies["Chem"].GetPeriodicTable().GetRvdw(int(number)) for number in numbers
    ], dtype=float)
    if not dependencies["np"].isfinite(radii).all() or (radii <= 0).any():
        raise ValueError("A positive RDKit van der Waals radius is required for every atom.")
    return radii


def _sasa_features(site: _DftSite, coordinates: Any, radii: Any, np: Any,
                   config: DescriptorExtractionConfig) -> dict[str, float]:
    count = config.sasa_points_per_atom
    index = np.arange(count, dtype=float) + 0.5
    z = 1.0 - 2.0 * index / count
    phi = index * (math.pi * (3.0 - math.sqrt(5.0)))
    xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directions = np.column_stack((xy * np.cos(phi), xy * np.sin(phi), z))
    expanded = radii + config.sasa_probe_radius_angstrom
    area = 0.0
    for atom_index in site.carboxyl_atoms:
        points = coordinates[atom_index] + expanded[atom_index] * directions
        exposed = np.ones(count, dtype=bool)
        center_distances = np.linalg.norm(coordinates - coordinates[atom_index], axis=1)
        blockers = np.where((center_distances < expanded + expanded[atom_index]) &
                            (np.arange(len(radii)) != atom_index))[0]
        for blocker in blockers:
            exposed &= np.sum((points - coordinates[blocker]) ** 2, axis=1) >= expanded[blocker] ** 2
        area += 4.0 * math.pi * expanded[atom_index] ** 2 * float(exposed.mean())
    return {"cooh_local_sasa_angstrom2": area}


def _radical_inverse(indices: Any, base: int, np: Any) -> Any:
    indices = indices.copy()
    result = np.zeros(len(indices), dtype=float)
    denominator = float(base)
    while np.any(indices):
        result += (indices % base) / denominator
        indices //= base
        denominator *= base
    return result


def _buried_feature(site: _DftSite, coordinates: Any, radii: Any, np: Any,
                    config: DescriptorExtractionConfig) -> dict[str, float]:
    indices = np.arange(1, config.buried_samples + 1, dtype=np.int64)
    u, v, w = (_radical_inverse(indices, base, np) for base in (2, 3, 5))
    radius = config.buried_radius_angstrom * np.cbrt(u)
    z = 1.0 - 2.0 * v
    transverse = np.sqrt(1.0 - z * z)
    directions = np.column_stack((transverse * np.cos(2 * math.pi * w),
                                  transverse * np.sin(2 * math.pi * w), z))
    center = coordinates[site.carboxyl_carbon]
    points = center + radius[:, None] * directions
    occupied = np.zeros(len(points), dtype=bool)
    for atom in range(len(radii)):
        if atom in site.carboxyl_atoms:
            continue
        if np.linalg.norm(coordinates[atom] - center) >= config.buried_radius_angstrom + radii[atom]:
            continue
        occupied |= np.sum((points - coordinates[atom]) ** 2, axis=1) <= radii[atom] ** 2
    return {"cooh_buried_volume_percent": 100.0 * float(occupied.mean())}


def _frontier_features(fchk: Mapping[str, Any], np: Any) -> dict[str, float]:
    alpha = np.asarray(fchk["alpha_orbital_energies_hartree"], dtype=float)
    na, nb = int(fchk["number_of_alpha_electrons"]), int(fchk["number_of_beta_electrons"])
    if na != nb or na < 1 or na >= len(alpha):
        raise ValueError("Closed-shell occupied and virtual alpha orbitals are required.")
    beta = fchk.get("beta_orbital_energies_hartree")
    if beta is not None and not np.allclose(alpha, np.asarray(beta), rtol=0, atol=1e-6):
        raise ValueError("Unrestricted orbitals are unsupported by this neutral singlet extractor.")
    homo, lumo = float(alpha[na - 1]), float(alpha[na])
    if lumo < homo:
        raise ValueError("LUMO energy is below HOMO energy.")
    return {"homo_ev": homo * HARTREE_TO_EV, "lumo_ev": lumo * HARTREE_TO_EV,
            "homo_lumo_gap_ev": (lumo - homo) * HARTREE_TO_EV}


def _npa_charges(job: str, fchk: Mapping[str, Any], deps: Mapping[str, Any]) -> Any:
    np, numbers = deps["np"], fchk["atomic_numbers"]
    symbols = tuple(deps["Chem"].GetPeriodicTable().GetElementSymbol(int(z)) for z in numbers)
    # NPA fields in a checkpoint can be retained/custom additions. Require the
    # selected final log's combined-spin NPA table to establish provenance.
    table = parse_npa_table(job, natoms=len(numbers))
    if tuple(table["symbols"]) != symbols:
        raise ValueError("NPA element/order does not match fchk.")
    charges = np.asarray(table["charges"], dtype=float)
    if charges.shape != (len(numbers),) or not np.isfinite(charges).all():
        raise ValueError("NPA charges have inconsistent length or non-finite values.")
    if abs(float(charges.sum()) - int(fchk["charge"])) > 0.02:
        raise ValueError("NPA charges do not sum to the molecular charge.")
    return charges


def _npa_site_features(charges: Any, site: _DftSite, np: Any) -> dict[str, float]:
    return dict(zip(DESCRIPTOR_COLUMNS[:8], (
        charges[site.acidic_hydrogen], charges[site.hydroxyl_oxygen],
        charges[site.carbonyl_oxygen], charges[site.carboxyl_carbon], charges[site.ipso_carbon],
        float(charges[list(site.ortho_carbons)].mean()),
        float(charges[list(site.meta_carbons)].mean()), charges[site.para_carbon]
    )))


def _npa_features(job: str, site: _DftSite, fchk: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, float]:
    return _npa_site_features(_npa_charges(job, fchk, deps), site, deps["np"])


def _wiberg_features(job: str, site: _DftSite, natoms: int) -> dict[str, float]:
    return _wiberg_site_features(parse_wiberg_matrix(job, natoms=natoms), site)


def _wiberg_site_features(matrix: Any, site: _DftSite) -> dict[str, float]:
    return {"wiberg_oh": float(matrix[site.hydroxyl_oxygen, site.acidic_hydrogen]),
            "wiberg_co_single": float(matrix[site.carboxyl_carbon, site.hydroxyl_oxygen]),
            "wiberg_co_double": float(matrix[site.carboxyl_carbon, site.carbonyl_oxygen]),
            "wiberg_aryl_cooh": float(matrix[site.carboxyl_carbon, site.ipso_carbon])}


def _mean_site_features(
    sites: Sequence[_DftSite], operation: Any, columns: Sequence[str],
) -> dict[str, float]:
    """Equal-weight mean of complete local site values; never skip a failed site."""
    site_values = []
    for site_index, site in enumerate(sites):
        try:
            values = operation(site)
            if set(values) != set(columns):
                raise ValueError("Local feature group returned inconsistent columns.")
            site_values.append({column: _finite_scalar(values[column], column) for column in columns})
        except Exception as exc:
            raise ValueError(
                f"Aromatic COOH site {site_index} (fchk C index {site.carboxyl_carbon}) failed: {exc}"
            ) from exc
    if not site_values:
        raise ValueError("At least one mapped aromatic COOH is required for a site mean.")
    return {column: math.fsum(values[column] for values in site_values) / len(site_values)
            for column in columns}


def _stretch_features(modes: Mapping[str, Any], coordinates: Any, site: _DftSite,
                      np: Any, config: DescriptorExtractionConfig) -> tuple[dict[str, float], dict[str, Any]]:
    frequencies = np.asarray(modes["frequencies_cm1"], dtype=float)
    displacements = np.asarray(modes["displacements"], dtype=float)
    if displacements.shape != (len(frequencies), len(coordinates), 3):
        raise ValueError("Normal-mode displacement dimensions do not match the geometry.")
    if len(frequencies) != 3 * len(coordinates) - 6:
        raise ValueError("Complete 3N-6 normal modes are required for bond-stretch assignment.")
    if not np.isfinite(frequencies).all() or not np.isfinite(displacements).all():
        raise ValueError("Normal modes contain non-finite values.")
    denominator = np.sum(displacements ** 2, axis=(1, 2))
    results, assignments = {}, {}
    for column, left, right in (("oh_stretch_frequency_cm1", site.hydroxyl_oxygen, site.acidic_hydrogen),
                                ("co_double_stretch_frequency_cm1", site.carboxyl_carbon, site.carbonyl_oxygen)):
        vector = coordinates[right] - coordinates[left]
        vector /= np.linalg.norm(vector)
        projection = (displacements[:, right] - displacements[:, left]) @ vector
        scores = projection ** 2 / np.maximum(denominator, 1e-30)
        scores[(frequencies <= 0) | (denominator < 1e-20)] = -1
        index = int(np.argmax(scores))
        if scores[index] < config.min_stretch_projection:
            raise ValueError(f"No mode has sufficient bond-stretch projection for {column}.")
        results[column] = float(frequencies[index] * config.frequency_scale)
        assignments[column] = {"mode_index_0based": index, "projection": float(scores[index]),
                               "unscaled_cm1": float(frequencies[index])}
    return results, assignments


def calculate_descriptors(
    smiles: str, logfile_path: Path | str, fchkfile_path: Path | str,
    dependencies: Mapping[str, Any] | None = None, *,
    config: DescriptorExtractionConfig | None = None,
    density_cube_path: Path | str | None = None, esp_cube_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return 30 DFT feature columns plus explicit completeness/provenance metadata."""
    deps = dependencies or _import_dependencies()
    np = deps["np"]
    config = config or _default_config()
    try:
        job = select_final_gaussian_job(Path(logfile_path).read_text(encoding=config.log_encoding))
        fchk = parse_fchk(fchkfile_path)
        sites, coordinates = _prepare_geometry(smiles, fchk, deps, config)
        log_coordinates = _validate_log_fchk(job, fchk, deps, config)
    except DescriptorExtractionError:
        raise
    except Exception as exc:
        raise DescriptorExtractionError(f"Cannot safely pair the structure/log/fchk: {exc}") from exc

    result = {column: math.nan for column in DESCRIPTOR_COLUMNS}
    issues: dict[str, str] = {}
    site_indices = [{key: getattr(site, key) for key in (
        "carboxyl_carbon", "carbonyl_oxygen", "hydroxyl_oxygen", "acidic_hydrogen",
        "ipso_carbon", "ring_atoms", "ortho_carbons", "meta_carbons", "para_carbon",
        "substituted_ring_carbons")} for site in sites]
    for site, indices in zip(sites, site_indices):
        substituted = frozenset(site.substituted_ring_carbons)
        ortho = len(substituted.intersection(site.ortho_carbons))
        meta = len(substituted.intersection(site.meta_carbons))
        para = int(site.para_carbon in substituted)
        indices["substituent_position_counts"] = {
            "total": ortho + meta + para, "ortho": ortho, "meta": meta, "para": para}
    provenance: dict[str, Any] = {"log": str(Path(logfile_path).resolve()),
                                  "fchk": str(Path(fchkfile_path).resolve()),
                                  "job_selection": "last normally terminated job only",
                                  "npa_source": "selected final log combined-spin NPA table",
                                  "aromatic_cooh_site_count": len(sites),
                                  "local_atom_indices_by_site_0based": site_indices,
                                  "site_aggregation": "equal-weight arithmetic mean; all sites required",
                                  "ignored_aliphatic_cooh_count": len(_mapped_matches(
                                      _parse_smiles(smiles, deps), ACID_SMARTS, deps)) - len(sites)}
    if len(sites) == 1:
        provenance["local_atom_indices_0based"] = site_indices[0]

    def collect(columns: Sequence[str], operation: Any) -> None:
        try:
            values = operation()
            if set(values) != set(columns):
                raise ValueError("Feature group returned inconsistent columns.")
            checked = {column: _finite_scalar(values[column], column) for column in columns}
            result.update(checked)
        except Exception as exc:
            for column in columns:
                issues[column] = f"{type(exc).__name__}: {exc}"

    def npa_mean() -> dict[str, float]:
        charges = _npa_charges(job, fchk, deps)
        return _mean_site_features(sites, lambda site: _npa_site_features(charges, site, np),
                                   DESCRIPTOR_COLUMNS[:8])
    def wiberg_mean() -> dict[str, float]:
        matrix = parse_wiberg_matrix(job, natoms=len(coordinates))
        return _mean_site_features(sites, lambda site: _wiberg_site_features(matrix, site),
                                   DESCRIPTOR_COLUMNS[8:12])
    collect(DESCRIPTOR_COLUMNS[:8], npa_mean)
    collect(DESCRIPTOR_COLUMNS[8:12], wiberg_mean)
    collect(DESCRIPTOR_COLUMNS[14:17], lambda: _frontier_features(fchk, np))
    collect(("dipole_moment_debye",), lambda: {"dipole_moment_debye":
            float(np.linalg.norm(fchk["dipole_au"])) * DIPOLE_AU_TO_DEBYE})
    collect(("isotropic_polarizability_au",), lambda: {"isotropic_polarizability_au":
            float(np.trace(fchk["polarizability_au"]) / 3.0)})
    collect(DESCRIPTOR_COLUMNS[23:28], lambda: _mean_site_features(
        sites, lambda site: _geometry_features(site, coordinates, np), DESCRIPTOR_COLUMNS[23:28]))
    collect(("cooh_local_sasa_angstrom2",), lambda: _mean_site_features(
        sites, lambda site: _sasa_features(site, coordinates, _vdw_radii(
            fchk["atomic_numbers"], deps), np, config), ("cooh_local_sasa_angstrom2",)))
    collect(("cooh_buried_volume_percent",), lambda: _mean_site_features(
        sites, lambda site: _buried_feature(site, coordinates, _vdw_radii(
            fchk["atomic_numbers"], deps), np, config), ("cooh_buried_volume_percent",)))
    provenance["geometric_sampling"] = {"sasa_points_per_atom": config.sasa_points_per_atom,
        "sasa_probe_angstrom": config.sasa_probe_radius_angstrom,
        "buried_samples": config.buried_samples, "buried_radius_angstrom": config.buried_radius_angstrom,
        "radii": "RDKit vdW", "buried_center": "COOH carbon; reference COOH atoms excluded"}

    modes = None
    try:
        # Checkpoints may retain a Hessian from an earlier linked job. Require
        # matching frequency output in the selected final log before using it.
        log_modes = parse_normal_modes(job, natoms=len(coordinates))
        modes = log_modes
        mode_coordinates = log_coordinates
        if fchk.get("frequencies_cm1") is not None and fchk.get("displacements") is not None:
            if np.asarray(fchk["frequencies_cm1"]).shape != np.asarray(log_modes["frequencies_cm1"]).shape or not np.allclose(
                fchk["frequencies_cm1"], log_modes["frequencies_cm1"], rtol=0, atol=0.05
            ):
                raise ValueError("fchk modes do not match the final log; a retained old Hessian is unsafe.")
            modes = {"frequencies_cm1": fchk["frequencies_cm1"],
                     "displacements": fchk["displacements"],
                     "atomic_numbers": fchk["atomic_numbers"], "precision": "fchk"}
            mode_coordinates = coordinates
        if modes.get("atomic_numbers") is not None and not np.array_equal(modes["atomic_numbers"], fchk["atomic_numbers"]):
            raise ValueError("Normal-mode atom order differs from fchk.")
        if config.require_minimum and np.min(modes["frequencies_cm1"]) < -config.imaginary_frequency_tolerance_cm1:
            raise DescriptorExtractionError("A significant imaginary mode indicates a non-minimum structure.")
        assignments = []
        def site_stretch(site: _DftSite) -> dict[str, float]:
            stretch, assignment = _stretch_features(modes, mode_coordinates, site, np, config)
            assignments.append({"carboxyl_carbon_0based": site.carboxyl_carbon,
                                "assignments": assignment})
            return stretch
        result.update(_mean_site_features(sites, site_stretch, DESCRIPTOR_COLUMNS[28:30]))
        provenance["normal_modes"] = {"assignments_by_site": assignments,
            "scale": config.frequency_scale, "precision": modes.get("precision", "fchk"),
            "aggregation": "mean of independently assigned harmonic-mode frequencies"}
        if len(sites) == 1:
            provenance["normal_modes"]["assignments"] = assignments[0]["assignments"]
    except DescriptorExtractionError:
        raise
    except Exception as exc:
        for column in DESCRIPTOR_COLUMNS[28:30]:
            issues[column] = f"{type(exc).__name__}: {exc}"

    with tempfile.TemporaryDirectory(prefix="benzoic-dft-cube-") as scratch:
        generation_issue = None
        if density_cube_path is None and esp_cube_path is None:
            executable = config.cubegen_path or shutil.which("cubegen")
            if executable:
                try:
                    density_cube_path, esp_cube_path = generate_cube_pair(
                        executable, fchkfile_path, scratch,
                        file_stem=Path(logfile_path).stem, npts=config.cube_npts,
                        timeout_seconds=config.cube_timeout_seconds)
                    provenance["cubegen"] = str(executable)
                except Exception as exc:
                    generation_issue = str(exc)
        density = None
        try:
            if density_cube_path is None:
                raise ValueError(generation_issue or "Density CUBE is absent; provide a cube or a Gaussian cubegen executable.")
            density = read_cube(density_cube_path)
            validate_cube_geometry(density, fchk["atomic_numbers"], coordinates,
                                   atol_angstrom=config.geometry_tolerance_angstrom)
            collect(("molecular_volume_angstrom3",), lambda: {
                "molecular_volume_angstrom3": density_volume(density, density_isovalue=config.density_isovalue)})
            provenance["density_surface"] = {"isovalue_e_bohr3": config.density_isovalue,
                "local_esp_radius_angstrom": config.esp_local_radius_angstrom,
                "volume": "occupied density grid; discretized electron-density isosurface volume",
                "surface_esp": "linear grid-edge crossing samples; nearest-nucleus local patches",
                "mpi": "arithmetic mean of absolute ESP over all grid-edge isosurface intersections",
                "density_cube": str(Path(density_cube_path).resolve())}
        except Exception as exc:
            density = None
            issues["molecular_volume_angstrom3"] = f"{type(exc).__name__}: {exc}"
        surface_columns = (*DESCRIPTOR_COLUMNS[12:14], "mpi_hartree_per_e")
        try:
            if density is None or esp_cube_path is None:
                raise ValueError(generation_issue or "Matched density and ESP cubes are required for surface ESP.")
            esp = read_cube(esp_cube_path)
            values = calculate_density_surface_features_for_sites(
                density, esp, [(site.acidic_hydrogen, site.carbonyl_oxygen) for site in sites],
                density_isovalue=config.density_isovalue,
                local_radius_angstrom=config.esp_local_radius_angstrom)
            provenance["density_surface"]["esp_cube"] = str(Path(esp_cube_path).resolve())
            result["mpi_hartree_per_e"] = _finite_scalar(values["mpi_hartree_per_e"], "MPI")
            if len(values["site_features"]) != len(sites) or len(values["site_issues"]) != len(sites):
                raise ValueError("Surface results do not cover every aromatic COOH site.")
            provenance["density_surface"]["local_values_by_site"] = values["site_features"]
            provenance["density_surface"]["local_issues_by_site"] = values["site_issues"]
            for column in DESCRIPTOR_COLUMNS[12:14]:
                failures = [f"site {i} (fchk C index {sites[i].carboxyl_carbon}): "
                    f"{site_issues.get(column, 'local surface value unavailable')}"
                    for i, (site_values, site_issues) in enumerate(zip(
                        values["site_features"], values["site_issues"]))
                    if column not in site_values or column in site_issues]
                if failures:
                    issues[column] = "; ".join(failures)
                else:
                    result[column] = math.fsum(_finite_scalar(v[column], column)
                        for v in values["site_features"]) / len(sites)
        except Exception as exc:
            for column in surface_columns:
                result[column] = math.nan
                issues[column] = f"{type(exc).__name__}: {exc}"
    missing = [column for column in DESCRIPTOR_COLUMNS if not math.isfinite(float(result[column]))]
    if missing and config.missing_policy == "raise":
        raise DescriptorExtractionError("Missing/invalid DFT features: " + ", ".join(
            f"{column} ({issues.get(column, 'unavailable')})" for column in missing))
    result.update({"descriptor_status": "partial" if missing else "complete",
        "missing_descriptors": ";".join(missing),
        "descriptor_issues": json.dumps(issues, ensure_ascii=False, sort_keys=True),
        "descriptor_provenance": json.dumps(provenance, ensure_ascii=False, sort_keys=True)})
    return result


def _row_path(row: Any, column: str, base: Path) -> Path | None:
    value = str(row[column]).strip() if column in row else ""
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _extract_row(row: Any, config: DescriptorExtractionConfig, dependencies: Mapping[str, Any]) -> dict[str, Any]:
    name, smiles = str(row[config.name_column]).strip(), str(row[config.smiles_column]).strip()
    if not name or not smiles:
        raise DescriptorExtractionError("Name and SMILES must be nonempty.")
    if any(c in name for c in '*?[]/\\:') or name in {'.', '..'}:
        raise DescriptorExtractionError("Name cannot be used safely as a filename.")
    stem = config.file_stem_template.format(name=name)
    base = config.input_path.resolve().parent
    log = _row_path(row, config.log_column, base) or config.logdata_path / f"{stem}.log"
    fchk = _row_path(row, config.fchk_column, base) or config.fchkdata_path / f"{stem}.fchk"
    density = _row_path(row, config.density_cube_column, base)
    esp = _row_path(row, config.esp_cube_column, base)
    if density is None and config.density_cube_path is not None:
        density = config.density_cube_path / f"{log.stem}_density.cube"
    if esp is None and config.esp_cube_path is not None:
        esp = config.esp_cube_path / f"{log.stem}_esp.cube"
    for source in (log, fchk, density, esp):
        if source is None:
            continue
        if any(_same_file(source, output) for output in (config.output_path, config.invalid_output_path)):
            raise _UnsafeOutputPath(f"An output would overwrite a calculation source file: {source}")
        if not source.is_file():
            raise DescriptorExtractionError(f"Calculation file was not found: {source}")
    return calculate_descriptors(smiles, log, fchk, dependencies, config=config,
        density_cube_path=density, esp_cube_path=esp)


def _read_input_csv(config: DescriptorExtractionConfig, pd: Any) -> Any:
    with config.input_path.open("r", encoding=config.input_encoding, newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next((record for record in reader if record), None)
        if not header or any(not column.strip() for column in header):
            raise ValueError("CSV must have a nonempty header with named columns.")
        if len(header) != len(set(header)):
            raise ValueError("Input CSV has duplicate columns.")
        for record in reader:
            if record and len(record) != len(header):
                raise ValueError(f"Malformed CSV record ending at line {reader.line_num}.")
    frame = pd.read_csv(config.input_path, encoding=config.input_encoding, dtype=str,
                        keep_default_na=False, index_col=False)
    if list(frame.columns) != header:
        raise ValueError("CSV columns were not preserved exactly.")
    missing = [column for column in (config.name_column, config.smiles_column) if column not in frame]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    conflicts = set(header) & (set(DESCRIPTOR_COLUMNS) | set(METADATA_COLUMNS) |
                               {"source_index", "error_type", "error_message"})
    if conflicts:
        raise ValueError(f"Input already contains reserved output columns: {sorted(conflicts)}")
    return frame


def extract_descriptors(config: DescriptorExtractionConfig | Mapping[str, Any]) -> DescriptorExtractionResult:
    if not isinstance(config, DescriptorExtractionConfig):
        config = DescriptorExtractionConfig(**dict(config))
    dependencies = _import_dependencies()
    pd = dependencies["pd"]
    frame = _read_input_csv(config, pd)
    valid_rows, invalid_rows, feature_rows = [], [], []
    counts: dict[str, int] = {}
    for source_index, row in frame.iterrows():
        try:
            values = _extract_row(row, config, dependencies)
        except _UnsafeOutputPath:
            raise
        except Exception as exc:
            if config.on_error == "raise":
                raise DescriptorExtractionError(f"Row {source_index} failed: {type(exc).__name__}: {exc}") from exc
            invalid = row.copy()
            invalid["source_index"], invalid["error_type"], invalid["error_message"] = source_index, type(exc).__name__, str(exc)
            invalid_rows.append(invalid)
            counts[type(exc).__name__] = counts.get(type(exc).__name__, 0) + 1
            continue
        valid = row.copy()
        valid["source_index"] = source_index
        valid_rows.append(valid)
        feature_rows.append(values)
    source_columns = list(frame.columns) + ["source_index"]
    output = pd.concat([pd.DataFrame(valid_rows, columns=source_columns).reset_index(drop=True),
                        pd.DataFrame(feature_rows, columns=DESCRIPTOR_COLUMNS + METADATA_COLUMNS)], axis=1)
    invalid = pd.DataFrame(invalid_rows, columns=source_columns + ["error_type", "error_message"]).reset_index(drop=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.invalid_output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.output_path, index=False, encoding=config.output_encoding)
    invalid.to_csv(config.invalid_output_path, index=False, encoding=config.output_encoding)
    return DescriptorExtractionResult(config, output, invalid, len(frame), len(output), counts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gaussian log/fchkから安息香酸誘導体のDFT特徴量30個を抽出")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--log-dir", type=Path, default=Path("."))
    parser.add_argument("--fchk-dir", type=Path, default=None)
    parser.add_argument("--density-cube-path", "--density-cube-dir", dest="density_cube_path",
                        type=Path, default=None)
    parser.add_argument("--esp-cube-path", "--esp-cube-dir", dest="esp_cube_path",
                        type=Path, default=None)
    parser.add_argument("--invalid-output", type=Path, default=None)
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--stem-template", default="sub_{name}")
    parser.add_argument("--on-error", choices=sorted(_ERROR_POLICIES), default="continue")
    parser.add_argument("--missing-policy", choices=("nan", "raise"), default="nan")
    parser.add_argument("--cubegen", default=None)
    parser.add_argument("--cube-npts", type=int, default=100)
    parser.add_argument("--density-isovalue", type=float, default=0.001)
    parser.add_argument("--frequency-scale", type=float, default=1.0)
    parser.add_argument("--allow-imaginary", action="store_true")
    parser.add_argument("--input-encoding", default="utf-8-sig")
    parser.add_argument("--output-encoding", default="utf-8-sig")
    parser.add_argument("--log-encoding", default="utf-8")
    args = parser.parse_args(argv)
    config = DescriptorExtractionConfig(args.input_csv, args.output_csv,
        logdata_path=args.log_dir, fchkdata_path=args.fchk_dir,
        density_cube_path=args.density_cube_path, esp_cube_path=args.esp_cube_path,
        invalid_output_path=args.invalid_output, name_column=args.name_column,
        smiles_column=args.smiles_column, file_stem_template=args.stem_template,
        on_error=args.on_error, missing_policy=args.missing_policy,
        cubegen_path=args.cubegen, cube_npts=args.cube_npts,
        density_isovalue=args.density_isovalue, frequency_scale=args.frequency_scale,
        require_minimum=not args.allow_imaginary, input_encoding=args.input_encoding,
        output_encoding=args.output_encoding, log_encoding=args.log_encoding)
    result = extract_descriptors(config)
    print(f"Read {result.rows_read}; valid {result.rows_written}; partial {result.rows_partial}; invalid {result.rows_excluded}")
    print(f"Valid: {config.output_path}\nInvalid: {config.invalid_output_path}")
    return 0


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



import math
import re
from pathlib import Path
from typing import Any

BOHR_TO_ANGSTROM = 0.529177210903


class GaussianParseError(ValueError):
    """Requested Gaussian data are missing, malformed, incomplete or ambiguous."""


def _numpy() -> Any:
    import numpy as np
    return np


def _gaussian_float(token: str, description: str = "Gaussian value") -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise GaussianParseError(f"Invalid {description}: {token!r}") from exc
    if not math.isfinite(value):
        raise GaussianParseError(f"Non-finite {description}: {token!r}")
    return value


def _fchk_header(line: str) -> tuple[str, str, str] | None:
    # Gaussian uses A40,3X,A1, so the type normally occurs at index43. Reading
    # the type from the fixed-label remainder also tolerates unpadded variants.
    if len(line) < 42 or not line[:40].strip():
        return None
    match = re.fullmatch(r"\s*([IRCLH])\s+(.+?)\s*", line[40:])
    if match is None:
        return None
    return line[:40].rstrip(), match[1], match[2]


def parse_fchk(path: Path | str, *, encoding: str = "utf-8") -> dict[str, Any]:
    return parse_fchk_text(Path(path).read_text(encoding=encoding, errors="strict"))


def parse_fchk_text(text: str) -> dict[str, Any]:
    """Read all labeled fchk fields and normalize commonly needed quantities.

Returned keys include fields, atomic_numbers, coordinates_bohr,
coordinates_angstrom, charge, multiplicity, number_of_alpha_electrons,
number_of_beta_electrons, alpha_orbital_energies_hartree,
beta_orbital_energies_hartree, dipole_au, polarizability_au, atomic_masses_u,
frequencies_cm1, and displacements. Optional missing properties are None.
"""
    np = _numpy()
    lines = text.splitlines()
    if len(lines) < 3:
        raise GaussianParseError("The fchk file lacks its title, method and data records.")
    fields: dict[str, Any] = {}
    normalized_labels: dict[str, str] = {}
    field_types: dict[str, str] = {}
    index = 2
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        header = _fchk_header(line)
        if header is None:
            raise GaussianParseError(f"Malformed fchk field header at line {index + 1}.")
        label, dtype, payload = header
        key = " ".join(label.casefold().split())
        previous_label = normalized_labels.get(key)
        index += 1
        count_match = re.fullmatch(r"N=\s*(\d+)", payload)
        if payload.startswith("N=") and count_match is None:
            raise GaussianParseError(f"Invalid array count for fchk field {label}.")
        if count_match is None:
            if dtype == "I":
                try:
                    value: Any = int(payload)
                except ValueError as exc:
                    raise GaussianParseError(f"Invalid integer field {label}.") from exc
            elif dtype == "R":
                value = _gaussian_float(payload, label)
            elif dtype == "L":
                if payload.upper() not in {"T", "F", ".TRUE.", ".FALSE."}:
                    raise GaussianParseError(f"Invalid logical field {label}.")
                value = payload.upper() in {"T", ".TRUE."}
            else:
                value = payload
            if previous_label is not None:
                if field_types[key] != dtype or fields[previous_label] != value:
                    raise GaussianParseError(
                        f"Conflicting duplicate fchk field: {label}."
                    )
            else:
                normalized_labels[key] = label
                field_types[key] = dtype
                fields[label] = value
            continue
        count = int(count_match[1])
        values: list[Any] = []
        while len(values) < count:
            if index >= len(lines) or _fchk_header(lines[index]) is not None:
                raise GaussianParseError(f"Truncated fchk array {label}: expected {count} values.")
            data_line = lines[index]
            index += 1
            if dtype in {"I", "R"}:
                tokens = data_line.split()
                try:
                    values.extend(
                        int(token) if dtype == "I" else _gaussian_float(token, label)
                        for token in tokens
                    )
                except ValueError as exc:
                    raise GaussianParseError(f"Malformed numeric fchk array {label}.") from exc
            elif dtype == "L":
                logicals = data_line.strip().replace(" ", "")
                if any(value.upper() not in {"T", "F"} for value in logicals):
                    raise GaussianParseError(f"Malformed logical fchk array {label}.")
                values.extend(value.upper() == "T" for value in logicals)
            else:
                # C arrays use 5A12; H arrays use 9A8. Blank strings are values.
                width, per_line = (12, 5) if dtype == "C" else (8, 9)
                remaining = min(count - len(values), per_line)
                padded = data_line.ljust(remaining * width)
                values.extend(padded[start * width:(start + 1) * width].rstrip()
                              for start in range(remaining))
            if len(values) > count:
                raise GaussianParseError(f"Too many values in fchk array {label}.")
        value = np.asarray(
            values, dtype={"I": int, "R": float, "L": bool}.get(dtype, object)
        )
        if previous_label is not None:
            if field_types[key] != dtype or not np.array_equal(fields[previous_label], value):
                raise GaussianParseError(
                    f"Conflicting duplicate fchk field: {label}."
                )
        else:
            normalized_labels[key] = label
            field_types[key] = dtype
            fields[label] = value

    def field(label: str, required: bool = False) -> Any:
        original = normalized_labels.get(" ".join(label.casefold().split()))
        if original is None:
            if required:
                raise GaussianParseError(f"Required fchk field is missing: {label}.")
            return None
        return fields[original]

    natoms = field("Number of atoms", True)
    if not isinstance(natoms, int) or natoms <= 0:
        raise GaussianParseError("Invalid Number of atoms in fchk.")
    raw_atomic_numbers = np.asarray(field("Atomic numbers", True))
    if raw_atomic_numbers.dtype.kind not in "iu":
        raise GaussianParseError("fchk Atomic numbers must be an integer array.")
    atomic_numbers = np.asarray(raw_atomic_numbers, dtype=int)
    coordinates = np.asarray(field("Current Cartesian coordinates", True), dtype=float)
    if atomic_numbers.shape != (natoms,) or coordinates.shape != (3 * natoms,):
        raise GaussianParseError("fchk atom numbers or Cartesian coordinate count is inconsistent.")
    if np.any(atomic_numbers <= 0) or np.any(atomic_numbers > 118):
        raise GaussianParseError("fchk contains ghost/dummy/invalid atomic numbers.")
    coordinates = coordinates.reshape(natoms, 3)
    charge, multiplicity = field("Charge", True), field("Multiplicity", True)
    if not isinstance(charge, int) or not isinstance(multiplicity, int) or multiplicity < 1:
        raise GaussianParseError("Invalid fchk charge or multiplicity.")

    def optional_array(label: str, length: int | None = None) -> np.ndarray | None:
        raw = field(label)
        if raw is None:
            return None
        result = np.asarray(raw, dtype=float)
        if result.ndim != 1 or (length is not None and len(result) != length):
            raise GaussianParseError(f"Invalid array shape for fchk {label}.")
        return result

    dipole = optional_array("Dipole Moment", 3)
    polarizability = optional_array("Polarizability", 6)
    if polarizability is not None:
        xx, xy, yy, xz, yz, zz = polarizability
        polarizability = np.asarray([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    nalpha, nbeta = field("Number of alpha electrons"), field("Number of beta electrons")
    for label, value in (("alpha", nalpha), ("beta", nbeta)):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise GaussianParseError(f"Invalid number of {label} electrons in fchk.")
    nelectrons = field("Number of electrons")
    if nalpha is not None and nbeta is not None:
        if nelectrons is not None and nelectrons != nalpha + nbeta:
            raise GaussianParseError("fchk alpha/beta counts disagree with Number of electrons.")
        if nalpha < nbeta or nalpha - nbeta != multiplicity - 1:
            raise GaussianParseError("fchk alpha/beta electron counts disagree with multiplicity.")
    alpha_energies = optional_array("Alpha Orbital Energies")
    beta_energies = optional_array("Beta Orbital Energies")
    for label, energies, occupied in (("alpha", alpha_energies, nalpha), ("beta", beta_energies, nbeta)):
        if energies is not None and occupied is not None and occupied > len(energies):
            raise GaussianParseError(f"fchk {label} occupied-electron count exceeds orbital count.")
    masses = optional_array("Real atomic weights", natoms)
    if masses is not None and np.any(masses <= 0):
        raise GaussianParseError("fchk atomic masses must be positive.")
    frequencies = displacements = None
    nmode = field("Number of Normal Modes")
    vib_e2, vib_modes = optional_array("Vib-E2"), optional_array("Vib-Modes")
    if nmode is not None:
        if not isinstance(nmode, int) or nmode < 1:
            raise GaussianParseError("Invalid fchk Number of Normal Modes.")
        if vib_e2 is not None:
            if len(vib_e2) < nmode:
                raise GaussianParseError("Truncated fchk Vib-E2 frequencies.")
            frequencies = vib_e2[:nmode].copy()
        if vib_modes is not None:
            if len(vib_modes) != nmode * natoms * 3:
                raise GaussianParseError("Invalid fchk Vib-Modes displacement count.")
            displacements = vib_modes.reshape(nmode, natoms, 3)
    elif vib_e2 is not None or vib_modes is not None:
        raise GaussianParseError("fchk vibration arrays lack Number of Normal Modes.")
    return {
        "title": lines[0].rstrip(), "header_lines": tuple(lines[:2]),
        "job_type": lines[1][:10].strip(),
        "method": lines[1][10:40].strip(), "basis": lines[1][40:70].strip(),
        "fields": fields, "atomic_numbers": atomic_numbers,
        "coordinates_bohr": coordinates,
        "coordinates_angstrom": coordinates * BOHR_TO_ANGSTROM,
        "charge": charge, "multiplicity": multiplicity,
        "total_energy_hartree": field("Total Energy"),
        "number_of_alpha_electrons": nalpha, "number_of_beta_electrons": nbeta,
        "alpha_orbital_energies_hartree": alpha_energies,
        "beta_orbital_energies_hartree": beta_energies,
        "dipole_au": dipole, "polarizability_au": polarizability,
        "atomic_masses_u": masses,
        "frequencies_cm1": frequencies, "displacements": displacements,
    }


def select_final_gaussian_job(text: str) -> str:
    """Select only the final calculation/internal Link1 step, requiring success.

Starts recognized: internal-job Link1 markers and full Gaussian launch markers.
A trailing started/failed job is an error even if an earlier job succeeded.
"""
    lines = text.splitlines(keepends=True)
    starts = [0]
    for index, line in enumerate(lines):
        if re.search(r"Link1:\s*Proceeding to internal job step number", line, re.I):
            starts.append(index)
        elif re.search(r"^\s*Entering Gaussian System", line):
            starts.append(index)
    start = max(starts)
    # Concatenated logs may omit the launch banner. In that case, a preceding
    # normal termination plus a new route is a safe additional boundary.
    normals = [index for index, line in enumerate(lines)
               if re.search(r"Normal termination of Gaussian", line)]
    if not normals:
        raise GaussianParseError("Gaussian log does not contain normal termination.")
    last_normal = normals[-1]
    if last_normal < start:
        raise GaussianParseError("The final Gaussian job is unfinished or failed.")
    trailing = "".join(lines[last_normal + 1:])
    if re.search(r"Error termination|Entering Gaussian|Link1:|^\s*#|SCF Done:|NAtoms=", trailing, re.M | re.I):
        raise GaussianParseError("A failed or unfinished Gaussian job follows the last normal termination.")
    if len(normals) > 1:
        previous_normal = normals[-2]
        if previous_normal >= start:
            start = previous_normal + 1
    selected = "".join(lines[start:last_normal + 1])
    if re.search(r"Error termination", selected, re.I):
        raise GaussianParseError("The selected Gaussian job contains error termination.")
    return selected


def _combined_nbo_markers(text: str, marker: str) -> tuple[list[str], list[int]]:
    lines = text.splitlines()
    spin_section = False
    starts: list[int] = []
    for index, line in enumerate(lines):
        compact = re.sub(r"\s+", "", line).upper()
        if "NATURALATOMICORBITAL" in compact:
            spin_section = False
        elif re.search(r"\b(?:Alpha|Beta) spin orbitals\b", line, re.I):
            spin_section = True
        if marker in line and not spin_section:
            starts.append(index)
    return lines, starts


def parse_npa_table(text: str, natoms: int | None = None) -> dict[str, Any]:
    """Parse the final complete combined-spin NPA summary, never a spin table."""
    np = _numpy()
    lines, starts = _combined_nbo_markers(text, "Summary of Natural Population Analysis")
    if not starts:
        raise GaussianParseError("No combined-spin Natural Population Analysis summary was found.")
    charges: dict[int, float] = {}
    symbols: dict[int, str] = {}
    complete = False
    for line in lines[starts[-1] + 1:]:
        row = re.match(r"^\s*([A-Z][a-z]?)\s+(\d+)\s+(\S+)", line)
        if row:
            atom = int(row[2]) - 1
            if atom < 0 or atom in charges:
                raise GaussianParseError("NPA table contains invalid or duplicate atom indices.")
            charges[atom] = _gaussian_float(row[3], "NPA charge")
            symbols[atom] = row[1]
        elif charges and re.match(r"^\s*\*?\s*Total\b", line, re.I):
            complete = True
            break
        elif charges and line.strip() and not re.fullmatch(r"[\s=\-*]+", line):
            break
    count = natoms if natoms is not None else len(charges)
    if not complete or count <= 0 or set(charges) != set(range(count)):
        raise GaussianParseError("The final NPA table is incomplete or has inconsistent atom indices.")
    return {"charges": np.asarray([charges[i] for i in range(count)]),
            "symbols": tuple(symbols[i] for i in range(count))}


def parse_npa_charges(text: str, natoms: int | None = None) -> dict[int, float]:
    table = parse_npa_table(text, natoms)
    return {index: float(charge) for index, charge in enumerate(table["charges"])}


def parse_wiberg_matrix(text: str, natoms: int | None = None) -> Any:
    """Parse blocked NAO Wiberg data, irrespective of row/column orientation."""
    np = _numpy()
    lines, starts = _combined_nbo_markers(text, "Wiberg bond index matrix in the NAO basis:")
    if not starts:
        raise GaussianParseError("No combined-spin NAO Wiberg bond-index matrix was found.")
    values: dict[tuple[int, int], float] = {}
    columns: tuple[int, ...] = ()
    complete = False
    for line in lines[starts[-1] + 1:]:
        if "Wiberg bond index, Totals by atom:" in line:
            complete = True
            break
        header = re.fullmatch(r"\s*Atom\s+([\d\s]+)", line)
        if header:
            columns = tuple(int(value) - 1 for value in header[1].split())
            if len(columns) != len(set(columns)) or any(column < 0 for column in columns):
                raise GaussianParseError("Invalid Wiberg matrix column header.")
            continue
        row = re.match(r"^\s*(\d+)\.?\s+([A-Z][a-z]?)\s+(.+)$", line)
        if row:
            atom = int(row[1]) - 1
            entries = row[3].split()
            if atom < 0 or not columns or len(entries) != len(columns):
                raise GaussianParseError("Malformed Wiberg blocked-matrix row.")
            for column, token in zip(columns, entries):
                key = (atom, column)
                value = _gaussian_float(token, "Wiberg bond index")
                if value < -1e-8:
                    raise GaussianParseError("Wiberg bond index must be nonnegative.")
                if key in values and not math.isclose(values[key], value, abs_tol=1e-8):
                    raise GaussianParseError("Conflicting repeated Wiberg matrix entry.")
                values[key] = value
        elif values and line.strip() and not re.fullmatch(r"[\s=\-]+", line):
            raise GaussianParseError("The final Wiberg table ended before its totals marker.")
    if not complete or not values:
        raise GaussianParseError("The final Wiberg matrix is incomplete.")
    inferred = max(max(pair) for pair in values) + 1
    count = natoms if natoms is not None else inferred
    if count < 1 or inferred != count:
        raise GaussianParseError("Wiberg matrix size does not match the molecular atom count.")
    matrix = np.full((count, count), np.nan)
    for (row, column), value in values.items():
        matrix[row, column] = value
    # Accommodate triangular print styles without assuming which triangle exists.
    missing = np.isnan(matrix)
    matrix[missing] = matrix.T[missing]
    if not np.all(np.isfinite(matrix)):
        raise GaussianParseError("Wiberg matrix has missing atom pairs.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.1e-4):
        raise GaussianParseError("Wiberg matrix is inconsistent with a symmetric bond index.")
    return (matrix + matrix.T) / 2.0


def last_standard_orientation(text: str) -> dict[str, Any]:
    """Return the final complete standard-orientation table in Angstroms."""
    np = _numpy()
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if "Standard orientation:" in line]
    if not starts:
        raise GaussianParseError("No Standard orientation table was found.")
    start = starts[-1]
    atom_numbers: list[int] = []
    coordinates: list[list[float]] = []
    complete = False
    angstrom_units = False
    for line in lines[start + 1:]:
        if re.search(r"Coordinates\s*\(Angstroms\)", line, re.I):
            angstrom_units = True
        row = re.fullmatch(r"\s*(\d+)\s+(-?\d+)\s+(-?\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s*", line)
        if row:
            if int(row[1]) != len(atom_numbers) + 1 or not 1 <= int(row[2]) <= 118:
                raise GaussianParseError("Invalid atom order/numbers in standard orientation.")
            atom_numbers.append(int(row[2]))
            coordinates.append([_gaussian_float(row[i], "Cartesian coordinate") for i in (4, 5, 6)])
        elif atom_numbers:
            if re.fullmatch(r"\s*-{5,}\s*", line):
                complete = True
            break
    if not complete:
        raise GaussianParseError("The final Standard orientation table is incomplete.")
    if not angstrom_units:
        raise GaussianParseError("Standard orientation lacks verified Angstrom units.")
    return {"atomic_numbers": np.asarray(atom_numbers, dtype=int),
            "coordinates_angstrom": np.asarray(coordinates, dtype=float)}


def last_cartesian_orientation(text: str) -> dict[str, Any]:
    """Use the final standard orientation, or input orientation for NoSymm logs.

The returned orientation name identifies the displacement coordinate frame.
An incomplete final standard table is an error rather than an input fallback.
"""
    if "Standard orientation:" in text:
        result = last_standard_orientation(text)
        result["orientation"] = "standard"
        return result
    starts = [match.start() for match in re.finditer(r"Input orientation:", text)]
    if not starts:
        raise GaussianParseError("No Standard/Input Cartesian orientation table was found.")
    selected = text[starts[-1]:].replace("Input orientation:", "Standard orientation:", 1)
    result = last_standard_orientation(selected)
    result["orientation"] = "input"
    return result


def parse_normal_modes(text: str, natoms: int | None = None) -> dict[str, Any]:
    """Parse the final harmonic analysis, including Freq=HPModes print layout.

Displacements have shape(mode,atom,xyz), in Gaussian's standard orientation.
If high precision and ordinary tables both exist, high precision is selected.
"""
    np = _numpy()
    lines = text.splitlines()
    if natoms is None:
        counts = re.findall(r"\bNAtoms=\s*(\d+)", text)
        if counts:
            natoms = int(counts[-1])
    analysis_starts = [index for index, line in enumerate(lines)
                       if re.search(r"Harmonic frequencies", line, re.I)]
    if analysis_starts:
        lines = lines[analysis_starts[-1]:]
    groups: dict[str, list[tuple[list[float], np.ndarray, np.ndarray]]] = {"standard": [], "high": []}
    frequency_re = re.compile(r"^\s*Frequencies\s*-{2,3}\s*(.*?)\s*$")
    index = 0
    while index < len(lines):
        match = frequency_re.match(lines[index])
        if match is None:
            index += 1
            continue
        frequencies = [_gaussian_float(token, "vibrational frequency") for token in match[1].split()]
        if not frequencies:
            raise GaussianParseError("An empty frequency block was found.")
        index += 1
        header = None
        while index < len(lines):
            tokens = lines[index].strip().split()
            if tokens[:3] == ["Atom", "AN", "X"]:
                header = "standard"
                break
            if tokens[:3] in (["Coord", "Atom", "Element:"], ["Coord", "Atom", "Element"]):
                header = "high"
                break
            if frequency_re.match(lines[index]) or re.search(r"Thermochemistry|Normal termination", lines[index]):
                break
            index += 1
        if header is None:
            raise GaussianParseError("Frequency block lacks Cartesian normal-mode displacements.")
        index += 1
        rows: dict[tuple[int, int], list[float]] = {}
        atomic_numbers: dict[int, int] = {}
        while index < len(lines):
            tokens = lines[index].split()
            required_prefix = 2 if header == "standard" else 3
            # A following block begins with its integer mode-number headings;
            # those headings are not atom rows when natoms was not supplied.
            if tokens and len(tokens) <= len(frequencies) and all(re.fullmatch(r"\d+", token) for token in tokens):
                break
            if len(tokens) < required_prefix or not all(re.fullmatch(r"\d+", token) for token in tokens[:required_prefix]):
                break
            if len(tokens) != required_prefix + (3 * len(frequencies) if header == "standard" else len(frequencies)):
                raise GaussianParseError("Frequency/displacement block column count is inconsistent.")
            if header == "standard":
                atom, atomic_number = int(tokens[0]) - 1, int(tokens[1])
                values = [_gaussian_float(token, "normal-mode displacement") for token in tokens[2:]]
                if (atom, 0) in rows:
                    raise GaussianParseError("Repeated atom in normal-mode displacement block.")
                rows[(atom, 0)] = values
            else:
                coordinate, atom, atomic_number = int(tokens[0]) - 1, int(tokens[1]) - 1, int(tokens[2])
                if coordinate not in {0, 1, 2} or (atom, coordinate) in rows:
                    raise GaussianParseError("Invalid/repeated Cartesian normal-mode coordinate.")
                rows[(atom, coordinate)] = [_gaussian_float(token, "normal-mode displacement") for token in tokens[3:]]
            if atom < 0 or not 1 <= atomic_number <= 118:
                raise GaussianParseError("Invalid atom index/element in normal-mode table.")
            if atom in atomic_numbers and atomic_numbers[atom] != atomic_number:
                raise GaussianParseError("Normal-mode table has inconsistent atom identities.")
            atomic_numbers[atom] = atomic_number
            index += 1
            if natoms is not None and len(rows) == natoms * (1 if header == "standard" else 3):
                break
        count = natoms if natoms is not None else len(atomic_numbers)
        if count < 1 or set(atomic_numbers) != set(range(count)):
            raise GaussianParseError("Normal-mode table is incomplete or has missing atoms.")
        displacements = np.empty((len(frequencies), count, 3))
        if header == "standard":
            for atom in range(count):
                displacements[:, atom, :] = np.asarray(rows[(atom, 0)]).reshape(len(frequencies), 3)
        else:
            if set(rows) != {(atom, coordinate) for atom in range(count) for coordinate in range(3)}:
                raise GaussianParseError("High-precision mode table has missing coordinates.")
            for atom in range(count):
                for coordinate in range(3):
                    displacements[:, atom, coordinate] = rows[(atom, coordinate)]
        elements = np.asarray([atomic_numbers[atom] for atom in range(count)], dtype=int)
        groups[header].append((frequencies, displacements, elements))
    precision = "high" if groups["high"] else "standard"
    blocks = groups[precision]
    if not blocks:
        raise GaussianParseError("No harmonic frequencies with displacement vectors were found.")
    reference = blocks[0][2]
    if any(not np.array_equal(block[2], reference) for block in blocks):
        raise GaussianParseError("Atom identities change between normal-mode blocks.")
    return {"frequencies_cm1": np.asarray([value for block in blocks for value in block[0]]),
            "displacements": np.concatenate([block[1] for block in blocks], axis=0),
            "atomic_numbers": reference, "precision": precision}



from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence
import math
import subprocess
import tempfile


CUBE_BOHR_TO_ANGSTROM = 0.529177210903


class CubeFeatureError(ValueError):
    """A scalar CUBE file or its requested molecular surface is invalid."""


@dataclass(frozen=True)
class CubeGrid:
    """Scalar grid with all coordinate fields converted to Angstroms."""

    origin_angstrom: Any
    axes_angstrom: Any
    atomic_numbers: Any
    atom_coordinates_angstrom: Any
    values: Any
    comments: tuple[str, str] = ("", "")
    source_path: str = ""


def _import_cube_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise CubeFeatureError("CUBE processing requires NumPy; install numpy.") from exc
    return np


def _cube_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def read_cube(
    path: str | Path,
    *,
    coordinate_unit: str = "bohr",
    max_grid_points: int = 2_000_000,
) -> CubeGrid:
    """Read one density/ESP scalar Gaussian CUBE without loading all text.

    Negative NATOMS (orbital datasets), NVAL != 1, truncated data, extra data,
    singular grids and nonfinite values are rejected.  Positive grid counts
    are required for ordinary Gaussian output.  Negative counts in an
    Angstrom-output dialect are accepted only with the explicit unit override.
    Density values remain electrons/Bohr**3 and ESP values Hartree/e; changing
    coordinate_unit does not change field-value units.
    """
    np = _import_cube_numpy()
    if coordinate_unit not in {"bohr", "angstrom"}:
        raise CubeFeatureError("coordinate_unit must be 'bohr' or 'angstrom'.")
    if not isinstance(max_grid_points, int) or isinstance(max_grid_points, bool) or max_grid_points < 8:
        raise CubeFeatureError("max_grid_points must be an integer >= 8.")
    source = Path(path)
    if not source.is_file():
        raise CubeFeatureError(f"CUBE file does not exist: {source}")
    try:
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            comments = (handle.readline().rstrip("\r\n"), handle.readline().rstrip("\r\n"))
            header = handle.readline().split()
            if len(header) not in (4, 5):
                raise CubeFeatureError("CUBE atom/origin header must contain 4 or 5 fields.")
            natoms = int(header[0])
            if natoms < 0:
                raise CubeFeatureError("Orbital/multiple-dataset CUBEs (negative NATOMS) are not density/ESP inputs.")
            if natoms == 0 or natoms > 10_000:
                raise CubeFeatureError("CUBE must contain 1..10000 atoms.")
            nval = int(header[4]) if len(header) == 5 else 1
            if nval != 1:
                raise CubeFeatureError("Only one scalar per grid point (NVAL=1) is supported.")
            origin = np.array([_cube_float(v) for v in header[1:4]], dtype=float)
            counts: list[int] = []
            axes: list[list[float]] = []
            for _ in range(3):
                row = handle.readline().split()
                if len(row) != 4:
                    raise CubeFeatureError("Each CUBE grid header must contain 4 fields.")
                signed_count = int(row[0])
                if signed_count < 0 and coordinate_unit == "bohr":
                    raise CubeFeatureError(
                        "Negative output grid counts are not standard Gaussian CUBE output. "
                        "For a verified Angstrom dialect, pass coordinate_unit='angstrom'."
                    )
                if abs(signed_count) < 2:
                    raise CubeFeatureError("Each CUBE axis must contain at least two grid points.")
                counts.append(abs(signed_count))
                axes.append([_cube_float(v) for v in row[1:]])
            npoints = math.prod(counts)
            if npoints > max_grid_points:
                raise CubeFeatureError(f"CUBE has {npoints} points, exceeding max_grid_points={max_grid_points}.")
            atomic_numbers = np.empty(natoms, dtype=int)
            coordinates = np.empty((natoms, 3), dtype=float)
            for atom_index in range(natoms):
                row = handle.readline().split()
                if len(row) != 5:
                    raise CubeFeatureError(f"Malformed CUBE atom row {atom_index + 1}.")
                atomic_number = int(row[0])
                if not 1 <= atomic_number <= 118:
                    raise CubeFeatureError("Dummy, ghost, and unknown nuclei are not supported in CUBEs.")
                if not math.isfinite(_cube_float(row[1])):
                    raise CubeFeatureError("CUBE nuclear charge is not finite.")
                atomic_numbers[atom_index] = atomic_number
                coordinates[atom_index] = [_cube_float(v) for v in row[2:]]
            # Preallocated float array avoids a list of millions of Python floats.
            flat_values = np.empty(npoints, dtype=float)
            offset = 0
            for line in handle:
                tokens = line.split()
                if not tokens:
                    continue
                if offset + len(tokens) > npoints:
                    raise CubeFeatureError("CUBE contains more field values than its grid specifies.")
                flat_values[offset:offset + len(tokens)] = [_cube_float(v) for v in tokens]
                offset += len(tokens)
            if offset != npoints:
                raise CubeFeatureError(f"Truncated CUBE data: expected {npoints} scalar values, got {offset}.")
            scale = CUBE_BOHR_TO_ANGSTROM if coordinate_unit == "bohr" else 1.0
            cube = CubeGrid(
                origin_angstrom=origin * scale,
                axes_angstrom=np.asarray(axes, dtype=float) * scale,
                atomic_numbers=atomic_numbers,
                atom_coordinates_angstrom=coordinates * scale,
                values=flat_values.reshape(tuple(counts)),
                comments=comments,
                source_path=str(source),
            )
    except (OSError, UnicodeError, ValueError, OverflowError) as exc:
        if isinstance(exc, CubeFeatureError):
            raise
        raise CubeFeatureError(f"Could not parse scalar CUBE {source}: {exc}") from exc
    _validate_cube_grid(cube)
    return cube


def _validate_cube_grid(cube: CubeGrid) -> None:
    np = _import_cube_numpy()
    values = np.asarray(cube.values)
    origin = np.asarray(cube.origin_angstrom)
    axes = np.asarray(cube.axes_angstrom)
    numbers = np.asarray(cube.atomic_numbers)
    coordinates = np.asarray(cube.atom_coordinates_angstrom)
    if values.ndim != 3 or min(values.shape) < 2:
        raise CubeFeatureError("CUBE values must be a 3D scalar grid, at least 2x2x2.")
    if origin.shape != (3,) or axes.shape != (3, 3):
        raise CubeFeatureError("Invalid CUBE origin or grid-step vectors.")
    if numbers.ndim != 1 or len(numbers) == 0 or coordinates.shape != (len(numbers), 3):
        raise CubeFeatureError("Invalid CUBE atom data.")
    if any(not np.isfinite(array).all() for array in (origin, axes, coordinates, values)):
        raise CubeFeatureError("CUBE grid, geometry, or scalar field contains nonfinite values.")
    if not np.issubdtype(numbers.dtype, np.integer) or np.any(numbers < 1) or np.any(numbers > 118):
        raise CubeFeatureError("CUBE atomic numbers must be integers in 1..118.")
    if abs(float(np.linalg.det(axes))) < 1e-12:
        raise CubeFeatureError("CUBE grid-step vectors have zero/singular voxel volume.")


def validate_cube_geometry(
    cube: CubeGrid,
    atomic_numbers: Any,
    coordinates_angstrom: Any,
    *,
    atol_angstrom: float = 1e-4,
) -> None:
    """Require a CUBE and fchk geometry to share atom order and orientation.

    No alignment, permutation or nearest-atom remapping is guessed.  A rotated
    or reordered CUBE must be regenerated from the same fchk before use.
    """
    np = _import_cube_numpy()
    if not np.array_equal(cube.atomic_numbers, np.asarray(atomic_numbers, dtype=int)):
        raise CubeFeatureError("CUBE and fchk atomic numbers/atom order differ.")
    reference = np.asarray(coordinates_angstrom, dtype=float)
    if reference.shape != cube.atom_coordinates_angstrom.shape or not np.allclose(
        cube.atom_coordinates_angstrom, reference, rtol=0.0, atol=atol_angstrom
    ):
        raise CubeFeatureError("CUBE and fchk coordinates/orientation differ.")


def _surface_point_chunks(
    density_cube: CubeGrid,
    esp_cube: CubeGrid,
    isovalue: float,
    *,
    slab_size: int = 16,
    point_chunk_size: int = 65536,
) -> Iterator[tuple[Any, Any]]:
    """Yield linearly interpolated density-isosurface grid-edge intersections."""
    np = _import_cube_numpy()
    rho = density_cube.values
    esp = esp_cube.values
    nx = rho.shape[0]
    for axis in range(3):
        x_end = nx - 1 if axis == 0 else nx
        for start in range(0, x_end, slab_size):
            stop = min(start + slab_size, x_end)
            if axis == 0:
                a, b = rho[start:stop], rho[start + 1:stop + 1]
                ea, eb = esp[start:stop], esp[start + 1:stop + 1]
            elif axis == 1:
                a, b = rho[start:stop, :-1, :], rho[start:stop, 1:, :]
                ea, eb = esp[start:stop, :-1, :], esp[start:stop, 1:, :]
            else:
                a, b = rho[start:stop, :, :-1], rho[start:stop, :, 1:]
                ea, eb = esp[start:stop, :, :-1], esp[start:stop, :, 1:]
            crossed = (a >= isovalue) != (b >= isovalue)
            index_tuple = np.nonzero(crossed)
            if not len(index_tuple[0]):
                continue
            fraction = (isovalue - a[index_tuple]) / (b[index_tuple] - a[index_tuple])
            indices = np.stack(index_tuple, axis=1).astype(float)
            indices[:, 0] += start
            indices[:, axis] += fraction
            points = density_cube.origin_angstrom + indices @ density_cube.axes_angstrom
            interpolated_esp = ea[index_tuple] + fraction * (eb[index_tuple] - ea[index_tuple])
            for offset in range(0, len(points), point_chunk_size):
                yield points[offset:offset + point_chunk_size], interpolated_esp[offset:offset + point_chunk_size]


def density_volume(cube: CubeGrid, density_isovalue: float = 0.001) -> float:
    """Occupied-grid-point volume (A**3), available without an ESP CUBE.

    The entire requested density isosurface must lie within the grid box.
    Finite resolution affects the occupied-point approximation; use consistent
    spacing/isovalues and assess convergence before comparing molecules.
    """
    np = _import_cube_numpy()
    _validate_cube_grid(cube)
    if not math.isfinite(density_isovalue) or density_isovalue <= 0:
        raise CubeFeatureError("density_isovalue must be positive and finite.")
    rho = cube.values
    if float(np.min(rho)) < -1e-8:
        raise CubeFeatureError("Density CUBE has negative values; a total electron density is required.")
    for boundary in (rho[0], rho[-1], rho[:, 0], rho[:, -1], rho[:, :, 0], rho[:, :, -1]):
        if np.any(boundary >= density_isovalue):
            raise CubeFeatureError("Density isosurface intersects the CUBE boundary; regenerate a larger box.")
    occupied_points = int(np.count_nonzero(rho >= density_isovalue))
    if occupied_points == 0:
        raise CubeFeatureError("Density CUBE contains no points at/above the requested isovalue.")
    voxel_volume = abs(float(np.linalg.det(cube.axes_angstrom)))
    return occupied_points * voxel_volume


def calculate_density_surface_features(
    density_cube: CubeGrid,
    esp_cube: CubeGrid,
    acidic_h_index: int,
    carbonyl_o_index: int,
    *,
    density_isovalue: float = 0.001,
    local_radius_angstrom: float = 2.0,
) -> dict[str, float]:
    """Return volume, local surface ESP, and molecular polarity index.

    Atom indices are zero-based Gaussian/fchk/CUBE atom indices.  Rejects
    mismatched grids or geometries, a truncated isosurface, wrong target atom
    elements, and a local region with no valid surface intersection.  The
    density is assumed to be a nonnegative total electron density in a.u.

    MPI is the arithmetic mean of |ESP| over all linearly interpolated grid-edge
    intersections with the requested electron-density isosurface. It is a
    discretized surface-sampling estimate in Hartree/e; compare molecules using
    identical density isovalues and grid resolution.
    """
    result = calculate_density_surface_features_for_sites(
        density_cube, esp_cube, [(acidic_h_index, carbonyl_o_index)],
        density_isovalue=density_isovalue,
        local_radius_angstrom=local_radius_angstrom,
    )
    if result["site_issues"][0]:
        # The single-site API historically raises when either local patch is
        # unavailable. Keep that behavior while multisite callers can retain
        # the global surface descriptors and report a missing local column.
        raise CubeFeatureError(next(iter(result["site_issues"][0].values())))
    return {
        "molecular_volume_angstrom3": result["molecular_volume_angstrom3"],
        **result["site_features"][0],
        "mpi_hartree_per_e": result["mpi_hartree_per_e"],
    }


def calculate_density_surface_features_for_sites(
    density_cube: CubeGrid,
    esp_cube: CubeGrid,
    site_atom_pairs: Sequence[tuple[int, int]],
    *,
    density_isovalue: float = 0.001,
    local_radius_angstrom: float = 2.0,
) -> dict[str, Any]:
    """Compute global surface descriptors and local ESP for every acid site.

    Each pair contains zero-based (acidic H, carbonyl O) CUBE atom indices.
    Returned ``site_features`` and ``site_issues`` lists preserve the number
    and order of input sites, including duplicate pairs. Each issue dictionary
    maps an unavailable local descriptor column to its error message; the
    corresponding value is omitted from that site's feature dictionary.

    Missing local patches do not discard MPI or volume. Invalid target atoms,
    incompatible grids/geometries, and incomplete density surfaces still
    raise CubeFeatureError. One surface traversal computes MPI and all local
    extrema; a shared target nucleus is evaluated only once.
    """
    np = _import_cube_numpy()
    _validate_cube_grid(density_cube)
    _validate_cube_grid(esp_cube)
    if not math.isfinite(density_isovalue) or density_isovalue <= 0:
        raise CubeFeatureError("density_isovalue must be positive and finite.")
    if not math.isfinite(local_radius_angstrom) or local_radius_angstrom <= 0:
        raise CubeFeatureError("local_radius_angstrom must be positive and finite.")
    if density_cube.values.shape != esp_cube.values.shape:
        raise CubeFeatureError("Density and ESP CUBE grid shapes differ.")
    for field in ("origin_angstrom", "axes_angstrom"):
        if not np.allclose(getattr(density_cube, field), getattr(esp_cube, field), rtol=0.0, atol=1e-6):
            raise CubeFeatureError(f"Density and ESP CUBE {field} differ; regenerate on the same grid.")
    validate_cube_geometry(
        esp_cube, density_cube.atomic_numbers, density_cube.atom_coordinates_angstrom,
        atol_angstrom=1e-5,
    )
    try:
        pairs = list(site_atom_pairs)
    except TypeError as exc:
        raise CubeFeatureError("site_atom_pairs must be a nonempty sequence of (acidic H, carbonyl O) pairs.") from exc
    if not pairs:
        raise CubeFeatureError("site_atom_pairs must contain at least one acid site.")
    targets: dict[int, int] = {}
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise CubeFeatureError("Each acid site must be an (acidic H, carbonyl O) index pair.")
        for index, element, label in ((pair[0], 1, "acidic H"), (pair[1], 8, "carbonyl O")):
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(density_cube.atomic_numbers):
                raise CubeFeatureError(f"Invalid zero-based CUBE index for {label}: {index}.")
            if int(density_cube.atomic_numbers[index]) != element:
                raise CubeFeatureError(f"The selected CUBE {label} has the wrong atomic number.")
            targets[index] = element
    volume = density_volume(density_cube, density_isovalue)
    extrema = {index: -math.inf if element == 1 else math.inf for index, element in targets.items()}
    counts = {index: 0 for index in targets}
    mpi_absolute_sum = 0.0
    mpi_sample_count = 0
    geometry = density_cube.atom_coordinates_angstrom
    radius_squared = local_radius_angstrom**2
    for points, potentials in _surface_point_chunks(density_cube, esp_cube, density_isovalue):
        mpi_absolute_sum += float(np.sum(np.abs(potentials), dtype=float))
        mpi_sample_count += len(potentials)
        # The nearest-nucleus distance is shared by all local patches. A small
        # tolerance preserves the previous behavior of accepting equidistant
        # intersections for each target instead of choosing an arbitrary atom.
        nearest_distance_squared = np.full(len(points), math.inf, dtype=float)
        for nucleus in geometry:
            distance_squared = np.sum((points - nucleus)**2, axis=1)
            np.minimum(nearest_distance_squared, distance_squared, out=nearest_distance_squared)
        for target_index, element in targets.items():
            target_distance_squared = np.sum((points - geometry[target_index])**2, axis=1)
            owned = (target_distance_squared <= radius_squared) & (
                nearest_distance_squared >= target_distance_squared - 1e-10
            )
            if np.any(owned):
                local_potentials = potentials[owned]
                if element == 1:
                    extrema[target_index] = max(extrema[target_index], float(np.max(local_potentials)))
                else:
                    extrema[target_index] = min(extrema[target_index], float(np.min(local_potentials)))
                counts[target_index] += int(np.count_nonzero(owned))
    if not mpi_sample_count:
        raise CubeFeatureError("The density isosurface has no grid-edge intersections.")
    site_features: list[dict[str, float]] = []
    site_issues: list[dict[str, str]] = []
    for h_index, o_index in pairs:
        features: dict[str, float] = {}
        issues: dict[str, str] = {}
        for index, label, column in (
            (h_index, "acidic H", "esp_max_acidic_h_hartree_per_e"),
            (o_index, "carbonyl O", "esp_min_carbonyl_o_hartree_per_e"),
        ):
            if counts[index]:
                features[column] = extrema[index]
            else:
                issues[column] = (
                    f"No density-isosurface intersection belongs to {label} within "
                    f"{local_radius_angstrom:g} Angstrom. Check atom mapping, radius, and grid resolution."
                )
        site_features.append(features)
        site_issues.append(issues)
    return {
        "molecular_volume_angstrom3": volume,
        "mpi_hartree_per_e": mpi_absolute_sum / mpi_sample_count,
        "site_features": site_features,
        "site_issues": site_issues,
    }


def generate_cube_pair(
    cubegen_path: str | Path,
    fchk_path: str | Path,
    output_dir: str | Path,
    *,
    file_stem: str | None = None,
    npts: int = 100,
    nprocs: int = 1,
    timeout_seconds: float = 600.0,
) -> tuple[Path, Path]:
    """Generate full SCF-density and same-grid SCF-ESP CUBEs in a unique folder.

    Uses Gaussian cubegen, supplied by the user's Gaussian installation.  The
    fchk and all existing files are left untouched.  The caller owns the
    returned files/subdirectory and may use a surrounding TemporaryDirectory
    for cleanup.  No shell commands or shell interpolation are used.  The
    cubegen executable is not downloaded or installed by this function.
    """
    if not isinstance(npts, int) or isinstance(npts, bool) or npts < 2:
        raise CubeFeatureError("cubegen npts must be an integer >= 2.")
    if not isinstance(nprocs, int) or isinstance(nprocs, bool) or nprocs < 0:
        raise CubeFeatureError("cubegen nprocs must be a nonnegative integer.")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise CubeFeatureError("cubegen timeout_seconds must be positive and finite.")
    fchk = Path(fchk_path).resolve()
    if not fchk.is_file():
        raise CubeFeatureError(f"fchk file does not exist: {fchk}")
    destination = Path(output_dir).resolve()
    output_stem = fchk.stem if file_stem is None else file_stem
    if (not isinstance(output_stem, str) or not output_stem
            or Path(output_stem).name != output_stem or output_stem in {".", ".."}):
        raise CubeFeatureError("cubegen file_stem must be a plain filename stem.")
    executable = str(cubegen_path)
    # An explicit relative executable path is resolved before cwd is changed;
    # a bare command name remains available for normal PATH lookup.
    if Path(executable).is_absolute() or Path(executable).parent != Path("."):
        executable = str(Path(executable).resolve())
    destination.mkdir(parents=True, exist_ok=True)
    generated_dir = Path(tempfile.mkdtemp(prefix="dft-cubes-", dir=str(destination)))
    density_path = generated_dir / f"{output_stem}_density.cube"
    esp_path = generated_dir / f"{output_stem}_esp.cube"
    # cubefile2 + npts=-1 makes the ESP cube use exactly the density cube grid.
    jobs = (
        [executable, str(nprocs), "FDensity=SCF", str(fchk), str(density_path), str(npts), "h"],
        [executable, str(nprocs), "Potential=SCF", str(fchk), str(esp_path), "-1", "h", str(density_path)],
    )
    for command, expected_path in zip(jobs, (density_path, esp_path)):
        try:
            completed = subprocess.run(
                command, cwd=str(generated_dir), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                errors="replace", shell=False, timeout=timeout_seconds, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CubeFeatureError(f"cubegen failed for {expected_path.name}: {exc}") from exc
        if completed.returncode != 0 or not expected_path.is_file() or expected_path.stat().st_size == 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise CubeFeatureError(
                f"cubegen failed for {expected_path.name} (exit={completed.returncode}): {detail}"
            )
    return density_path, esp_path


if __name__ == "__main__":
    raise SystemExit(main())
