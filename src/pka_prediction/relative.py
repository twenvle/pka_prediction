import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# データの読み込み
df = pd.read_csv(
    r"C:\Users\kyoma\Desktop\laboratory\ml\predict_pka\out\csvfile\benzoic_acid_features.csv"
)

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

X = df[descriptors]  # 記述子のみにする


# ==========================================
# 1. 相関係数の計算
# ==========================================
corr_matrix = X.corr()

# 数値で確認したい場合（相関が高いペアを表示）
print("【相関が高いペアのリスト (|r| > 0.8)】")
threshold = 0.8

# 重複を除いてループ
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        val = corr_matrix.iloc[i, j]
        if abs(val) > threshold:
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            print(f"{col1} - {col2}: {val:.4f}")

# ==========================================
# 2. ヒートマップの描画
# ==========================================
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()
