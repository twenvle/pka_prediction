import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# データの読み込み
df = pd.read_csv(
    r"D:\kikuchi\pka_prediction\data\processed\labeled\merged_dataset.csv"
)
"""
descriptors = [
    "pKa",
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
"""

descriptors = [
    "pka_value",
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
    "npa_acidic_h",
    "npa_hydroxyl_o",
    "npa_carbonyl_o",
    "npa_carboxyl_c",
    "npa_ipso_c",
    "npa_ortho_c_mean",
    "npa_meta_c_mean",
    "npa_para_c",
    "wiberg_oh",
    "wiberg_co_single",
    "wiberg_co_double",
    "wiberg_aryl_cooh",
    "esp_max_acidic_h_hartree_per_e",
    "esp_min_carbonyl_o_hartree_per_e",
    "homo_ev",
    "lumo_ev",
    "homo_lumo_gap_ev",
    "dipole_moment_debye",
    "isotropic_polarizability_au",
    "mpi_hartree_per_e",
    "molecular_volume_angstrom3",
    "cooh_local_sasa_angstrom2",
    "cooh_buried_volume_percent",
    "oh_bond_length_angstrom",
    "co_double_bond_length_angstrom",
    "co_single_bond_length_angstrom",
    "aryl_cooh_bond_length_angstrom",
    "benzene_cooh_plane_angle_deg",
    "oh_stretch_frequency_cm1",
    "co_double_stretch_frequency_cm1",
]

X = df[descriptors]  # 記述子のみにする


# 目的変数を除外して説明変数のみ取得
X = df[descriptors].drop(columns=["pka_value"])

# 相関係数を計算
corr_matrix = X.corr()

threshold = 0.8

# 残す記述子と除外する記述子
kept_descriptors = []
removed_descriptors = []

for col in X.columns:
    # すでに残した記述子との相関を確認
    high_corr = {
        kept: corr_matrix.loc[col, kept]
        for kept in kept_descriptors
        if abs(corr_matrix.loc[col, kept]) >= threshold
    }

    if high_corr:
        removed_descriptors.append(col)

        print(f"除外: {col}")
        for kept, corr in high_corr.items():
            print(f"  → {kept} との相関: {corr:.4f}")
    else:
        kept_descriptors.append(col)

# 除外後のDataFrame
X_filtered = X[kept_descriptors]

print("\n除外された記述子:")
print(removed_descriptors)
print("\n残された記述子:")
print(kept_descriptors)

print(f"\n除外前: {X.shape[1]} 個")
print(f"除外後: {X_filtered.shape[1]} 個")

# 除外後の相関ヒートマップ
plt.figure(figsize=(12, 10))
sns.heatmap(X_filtered.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap (Filtered)")
plt.show()
