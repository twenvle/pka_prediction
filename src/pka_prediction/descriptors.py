#!/usr/bin/env python3
"""CSVと対応するGaussianログからカルボン酸記述子を抽出する。

成功行は ``output_path`` へ、失敗行はエラー情報付きで
``invalid_output_path`` へ保存する。入力CSVの値は文字列として読み込み、
IDの先頭ゼロや文字列 ``NA`` を保持する。
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "DescriptorExtractionConfig",
    "DescriptorExtractionResult",
    "DescriptorExtractionError",
    "DESCRIPTOR_COLUMNS",
    "extract_descriptors",
]

ACID_SMARTS: str = "[CX3](=[OX1])[OX2H1][H]"
CarboxylSite = tuple[int, int, int, int]

DESCRIPTOR_COLUMNS = (
    "sasa",
    "polar",
    "oh_distance",
    "co_distance",
    "co_double_distance",
    "h_charge",
    "o_single_charge",
    "o_double_charge",
    "cooh_charge",
    "oh_bond",
    "co_bond",
    "co_double_bond",
    "homo_ev",
    "lumo_ev",
    "dipole_moment_debye",
    "logp",
    "hbd",
    "hba",
)
_ERROR_POLICIES = frozenset({"continue", "raise"})


def _same_file(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    return first.exists() and second.exists() and first.samefile(second)


@dataclass(frozen=True)
class DescriptorExtractionConfig:
    """CSV特徴量抽出の入出力条件。"""

    input_path: Path | str
    output_path: Path | str
    logdata_path: Path | str
    invalid_output_path: Path | str | None = None
    labeled: bool = False
    name_column: str = "name"
    smiles_column: str = "smiles"
    input_encoding: str = "utf-8-sig"
    output_encoding: str = "utf-8-sig"
    log_encoding: str = "utf-8"
    on_error: str = "continue"

    def __post_init__(self) -> None:
        if self.on_error not in _ERROR_POLICIES:
            raise ValueError("on_error must be 'continue' or 'raise'.")
        codecs.lookup(self.input_encoding)
        codecs.lookup(self.output_encoding)
        codecs.lookup(self.log_encoding)
        if not self.name_column.strip() or not self.smiles_column.strip():
            raise ValueError("name_column and smiles_column must not be empty.")
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_path", Path(self.output_path))
        object.__setattr__(self, "logdata_path", Path(self.logdata_path))
        invalid_output_path = self.invalid_output_path
        if invalid_output_path is None:
            suffix = self.output_path.suffix or ".csv"
            invalid_output_path = self.output_path.with_name(
                f"{self.output_path.stem}_invalid{suffix}"
            )
        object.__setattr__(self, "invalid_output_path", Path(invalid_output_path))
        if _same_file(self.input_path, self.output_path):
            raise ValueError(
                "入力ファイルと出力ファイルには異なるパスを指定してください"
            )
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
    """入力行にCAS、SMILES、logdata、計算上の問題がある場合の例外。"""


def _find_carboxyl_sites(
    smiles: str, dependencies: Mapping[str, Any]
) -> tuple[CarboxylSite, ...]:
    Chem = dependencies["Chem"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DescriptorExtractionError("SMILES could not be parsed by RDKit.")
    mol = Chem.AddHs(mol)

    carboxyl_pattern = Chem.MolFromSmarts(ACID_SMARTS)
    if carboxyl_pattern is None:
        raise RuntimeError(f"Internal SMARTS is invalid: {ACID_SMARTS}")
    raw_matches = mol.GetSubstructMatches(carboxyl_pattern, uniquify=True)
    if any(len(match) != 4 for match in raw_matches):
        raise RuntimeError("Internal carboxylic-acid SMARTS returned an invalid match.")
    matches: tuple[CarboxylSite, ...] = tuple(
        (int(match[0]), int(match[1]), int(match[2]), int(match[3]))
        for match in raw_matches
    )
    if not matches:
        raise DescriptorExtractionError("No carboxylic acid group was found.")
    return matches


def _import_dependencies() -> dict[str, Any]:
    try:
        import cclib
        import freesasa
        import numpy as np
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
    except ImportError as exc:
        raise RuntimeError(
            "Descriptor extraction requires cclib, freesasa, numpy, pandas and RDKit. "
            "Install the project requirements before running extraction."
        ) from exc
    return {
        "cclib": cclib,
        "freesasa": freesasa,
        "np": np,
        "pd": pd,
        "Chem": Chem,
        "Descriptors": Descriptors,
        "Lipinski": Lipinski,
    }


def _read_final_nbo_charges(
    logfile_path: Path, encoding: str = "utf-8"
) -> dict[int, float]:
    """最後のNatural Population Analysis表から1-based原子番号の電荷を読む。"""
    lines = logfile_path.read_text(encoding=encoding, errors="replace").splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if "Summary of Natural Population Analysis" in line
    ]
    if not starts:
        raise DescriptorExtractionError("Natural Population Analysis was not found.")
    charges: dict[int, float] = {}
    for line in lines[starts[-1] + 1 :]:
        if charges and ("Total" in line or "====" in line):
            break
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            atom_number = int(parts[1])
            charge = float(parts[2])
        except ValueError:
            continue
        charges[atom_number] = charge
    if not charges:
        raise DescriptorExtractionError(
            "Natural Population Analysis contained no charges."
        )
    return charges


def _wiberg_bond_index(
    lines: Sequence[str], start: int, site_x: int, site_y: int
) -> float:
    """Wiberg行列からRDKitの0-based原子間の値を読む。"""
    atom_x = site_x + 1
    atom_y = site_y + 1
    column_atom = min(atom_x, atom_y)
    row_atom = max(atom_x, atom_y)
    header_atoms: list[int] = []
    for line in lines[start + 1 :]:
        if "Wiberg bond index, Totals by atom:" in line:
            break
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Atom":
            header_atoms = []
            for token in parts[1:]:
                try:
                    header_atoms.append(int(token))
                except ValueError:
                    continue
            continue

        try:
            current_row = int(parts[0].rstrip("."))
        except ValueError:
            continue
        if current_row != row_atom or column_atom not in header_atoms:
            continue

        value_index = 2 + header_atoms.index(column_atom)
        if value_index >= len(parts):
            raise DescriptorExtractionError(
                f"Malformed Wiberg matrix row for atom {row_atom}."
            )
        try:
            return float(parts[value_index])
        except ValueError as exc:
            raise DescriptorExtractionError(
                f"Invalid Wiberg value for atoms {atom_x} and {atom_y}."
            ) from exc

    raise DescriptorExtractionError(
        f"Wiberg bond index was not found for atoms {atom_x} and {atom_y}."
    )


def calculate_final_nao(
    logfile_path: Path,
    sites: Sequence[CarboxylSite],
    encoding: str = "utf-8",
) -> dict[str, float]:
    h_sites = [site[3] for site in sites]
    o_single_sites = [site[2] for site in sites]
    o_double_sites = [site[1] for site in sites]
    c_sites = [site[0] for site in sites]
    lines = logfile_path.read_text(encoding=encoding, errors="replace").splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if "Wiberg bond index matrix in the NAO basis:" in line
    ]
    if not starts:
        raise DescriptorExtractionError("Natural Atomic Orbital was not found.")
    oh_bond, co_bond, co_double_bond = [], [], []
    for h_site, o_single_site, o_double_site, c_site in zip(
        h_sites, o_single_sites, o_double_sites, c_sites
    ):
        oh_bond.append(_wiberg_bond_index(lines, starts[-1], h_site, o_single_site))
        co_bond.append(_wiberg_bond_index(lines, starts[-1], c_site, o_single_site))
        co_double_bond.append(
            _wiberg_bond_index(lines, starts[-1], c_site, o_double_site)
        )
    return {
        "oh_bond": max(oh_bond),
        "co_bond": min(co_bond),
        "co_double_bond": min(co_double_bond),
    }


def calculate_nbo_descriptors(
    logfile_path: Path,
    sites: Sequence[CarboxylSite],
    encoding: str = "utf-8",
) -> dict[str, float]:
    charges = _read_final_nbo_charges(logfile_path, encoding)
    h_sites = [site[3] for site in sites]
    o_single_sites = [site[2] for site in sites]
    o_double_sites = [site[1] for site in sites]
    required_sites = {index for site in sites for index in site}
    missing = sorted(index + 1 for index in required_sites if index + 1 not in charges)
    if missing:
        raise DescriptorExtractionError(
            "Natural Population Analysis is missing atom numbers: "
            + ", ".join(map(str, missing))
        )

    def charge(site: int) -> float:
        return charges[site + 1]

    return {
        "h_charge": max(charge(site) for site in h_sites),
        "o_single_charge": min(charge(site) for site in o_single_sites),
        "o_double_charge": min(charge(site) for site in o_double_sites),
        "cooh_charge": sum(sum(charge(index) for index in site) for site in sites)
        / len(sites),
    }


def _bond_distance(logdata: Any, site_x: int, site_y: int) -> float:
    coordinates = logdata.atomcoords[-1]
    try:
        x_1, y_1, z_1 = (float(value) for value in coordinates[site_x])
        x_2, y_2, z_2 = (float(value) for value in coordinates[site_y])
    except (IndexError, TypeError, ValueError) as exc:
        raise DescriptorExtractionError(
            f"Coordinates were not found for atoms {site_x + 1} and {site_y + 1}."
        ) from exc
    return ((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2 + (z_2 - z_1) ** 2) ** 0.5


def calculate_bond_distance(
    logdata: Any, sites: Sequence[CarboxylSite]
) -> dict[str, float]:
    h_sites = [site[3] for site in sites]
    o_single_sites = [site[2] for site in sites]
    o_double_sites = [site[1] for site in sites]
    c_sites = [site[0] for site in sites]
    oh_distance, co_distance, co_double_distance = [], [], []
    for h_site, o_single_site, o_double_site, c_site in zip(
        h_sites, o_single_sites, o_double_sites, c_sites
    ):
        oh_distance.append(_bond_distance(logdata, h_site, o_single_site))
        co_distance.append(_bond_distance(logdata, c_site, o_single_site))
        co_double_distance.append(_bond_distance(logdata, c_site, o_double_site))

    return {
        "oh_distance": max(oh_distance),
        "co_distance": min(co_distance),
        "co_double_distance": min(co_double_distance),
    }


def calculate_frontier_orbitals(logdata: Any) -> dict[str, float]:
    homo_index = int(logdata.homos[0])
    homo_ev = float(logdata.moenergies[0][homo_index])
    lumo_ev = float(logdata.moenergies[0][homo_index + 1])
    return {
        "homo_ev": homo_ev,
        "lumo_ev": lumo_ev,
    }


def calculate_polarizability(logdata: Any, np: Any) -> dict[str, float]:
    return {"polar": float(np.trace(logdata.polarizabilities[-1]) / 3.0)}


def calculate_dipole_moment(logdata: Any, np: Any) -> dict[str, float]:
    return {"dipole_moment_debye": float(np.linalg.norm(logdata.moments[1]))}


def calculate_carboxyl_sasa(
    logdata: Any,
    carboxyl_groups: Sequence[CarboxylSite],
    dependencies: Mapping[str, Any],
) -> dict[str, float]:
    Chem = dependencies["Chem"]
    coordinates = [float(value) for row in logdata.atomcoords[-1] for value in row]
    periodic_table = Chem.GetPeriodicTable()
    radii = [periodic_table.GetRvdw(int(number)) for number in logdata.atomnos]
    result = dependencies["freesasa"].calcCoord(coordinates, radii)
    atom_areas = [
        float(result.atomArea(index)) for index in range(len(logdata.atomnos))
    ]
    group_areas = [
        sum(atom_areas[index] for index in group) for group in carboxyl_groups
    ]
    return {"sasa": float(sum(group_areas))}


def calculate_rdkit_descriptors(
    smiles: str, dependencies: Mapping[str, Any]
) -> dict[str, float | int]:
    mol = dependencies["Chem"].MolFromSmiles(smiles)
    if mol is None:
        raise DescriptorExtractionError("SMILES could not be parsed by RDKit.")
    return {
        "logp": float(dependencies["Descriptors"].MolLogP(mol)),
        "hbd": int(dependencies["Lipinski"].NumHDonors(mol)),
        "hba": int(dependencies["Lipinski"].NumHAcceptors(mol)),
    }


def _validate_atom_order(
    smiles: str, logdata: Any, dependencies: Mapping[str, Any]
) -> None:
    """RDKitとGaussian/cclibの原子列が一致することを確認する。"""
    Chem = dependencies["Chem"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DescriptorExtractionError("SMILES could not be parsed by RDKit.")
    mol = Chem.AddHs(mol)
    rdkit_atomnos = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    try:
        log_atomnos = [int(number) for number in logdata.atomnos]
    except (AttributeError, TypeError, ValueError) as exc:
        raise DescriptorExtractionError(
            "Atomic numbers could not be read from the Gaussian log."
        ) from exc
    if rdkit_atomnos != log_atomnos:
        raise DescriptorExtractionError(
            "RDKit and Gaussian atom ordering differ; descriptor indices are unsafe."
        )


def calculate_descriptors(
    smiles: str,
    logfile_path: Path,
    logdata: Any,
    sites: Sequence[CarboxylSite],
    dependencies: Mapping[str, Any],
    *,
    log_encoding: str = "utf-8",
) -> dict[str, Any]:
    """1分子についてすべての記述子を計算する。"""
    result: dict[str, Any] = {}
    result.update(
        calculate_carboxyl_sasa(
            logdata,
            sites,
            dependencies,
        )
    )
    result.update(calculate_polarizability(logdata, dependencies["np"]))
    result.update(calculate_bond_distance(logdata, sites))
    result.update(
        calculate_nbo_descriptors(
            logfile_path,
            sites,
            log_encoding,
        )
    )
    result.update(calculate_final_nao(logfile_path, sites, log_encoding))
    result.update(calculate_dipole_moment(logdata, dependencies["np"]))
    result.update(calculate_frontier_orbitals(logdata))
    result.update(calculate_rdkit_descriptors(smiles, dependencies))
    expected = set(DESCRIPTOR_COLUMNS)
    actual = set(result)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise DescriptorExtractionError(
            f"Descriptor columns are inconsistent; missing={missing}, "
            f"unexpected={unexpected}."
        )
    return {column: result[column] for column in DESCRIPTOR_COLUMNS}


def _find_logfile(name: str, config: DescriptorExtractionConfig) -> Path:
    if not name or any(character in name for character in "*?[]/\\"):
        raise DescriptorExtractionError(
            f"Invalid CAS value for a log filename: {name!r}"
        )
    pattern = (
        f"labeled/sub_{name}.log" if config.labeled else f"unlabeled/sub_{name}.log"
    )
    matches = sorted(
        path for path in config.logdata_path.glob(pattern) if path.is_file()
    )
    if not matches:
        raise DescriptorExtractionError(
            f"Log file was not found for {name}: {config.logdata_path / pattern}"
        )
    if len(matches) > 1:
        raise DescriptorExtractionError(
            f"Multiple log files were found for CAS {name}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _extract_row(
    row: Any,
    config: DescriptorExtractionConfig,
    dependencies: Mapping[str, Any],
) -> dict[str, Any]:
    name = str(row[config.name_column]).strip()
    smiles = str(row[config.smiles_column]).strip()
    if not name:
        raise DescriptorExtractionError("Name is empty.")
    if not smiles:
        raise DescriptorExtractionError("SMILES is empty.")
    sites = _find_carboxyl_sites(smiles, dependencies)
    logfile_path = _find_logfile(name, config)
    logdata = dependencies["cclib"].io.ccread(str(logfile_path))
    if logdata is None:
        raise DescriptorExtractionError(
            f"cclib could not parse the log file: {logfile_path}"
        )
    _validate_atom_order(smiles, logdata, dependencies)
    return calculate_descriptors(
        smiles,
        logfile_path,
        logdata,
        sites,
        dependencies,
        log_encoding=config.log_encoding,
    )


def extract_descriptors(
    config: DescriptorExtractionConfig | Mapping[str, Any],
) -> DescriptorExtractionResult:
    """CSVから記述子を抽出し、成功行と失敗行をそれぞれ保存する。"""
    if not isinstance(config, DescriptorExtractionConfig):
        config = DescriptorExtractionConfig(**dict(config))

    dependencies = _import_dependencies()
    pd = dependencies["pd"]
    dataframe = pd.read_csv(
        config.input_path,
        encoding=config.input_encoding,
        dtype=str,
        keep_default_na=False,
    )
    if not dataframe.columns.is_unique:
        raise ValueError("Input data must not contain duplicate column names.")
    missing = [
        column
        for column in (config.name_column, config.smiles_column)
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    reserved_columns = set(DESCRIPTOR_COLUMNS) | {
        "source_index",
        "error_type",
        "error_message",
    }
    conflicts = sorted(reserved_columns.intersection(dataframe.columns))
    if conflicts:
        raise ValueError(
            "Input data already contains output columns: " + ", ".join(conflicts)
        )

    valid_rows: list[Any] = []
    feature_rows: list[dict[str, Any]] = []
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
            error_counts[type(exc).__name__] = (
                error_counts.get(type(exc).__name__, 0) + 1
            )
            continue
        valid = row.copy()
        valid["source_index"] = source_index
        valid_rows.append(valid)
        feature_rows.append(descriptors)

    source_columns = list(dataframe.columns) + ["source_index"]
    valid_frame = pd.DataFrame(valid_rows, columns=source_columns).reset_index(
        drop=True
    )
    feature_frame = pd.DataFrame(feature_rows, columns=DESCRIPTOR_COLUMNS)
    output = pd.concat([valid_frame, feature_frame], axis=1)
    invalid_columns = list(dataframe.columns) + [
        "source_index",
        "error_type",
        "error_message",
    ]
    invalid_output = pd.DataFrame(invalid_rows, columns=invalid_columns).reset_index(
        drop=True
    )

    invalid_output_path = config.invalid_output_path
    if invalid_output_path is None:  # __post_init__で設定されるため通常は到達しない
        raise RuntimeError("invalid_output_path was not initialized.")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.output_path, index=False, encoding=config.output_encoding)
    invalid_output.to_csv(
        invalid_output_path,
        index=False,
        encoding=config.output_encoding,
    )

    return DescriptorExtractionResult(
        config=config,
        dataframe=output,
        invalid_dataframe=invalid_output,
        rows_read=len(dataframe),
        rows_written=len(output),
        error_counts=error_counts,
    )
