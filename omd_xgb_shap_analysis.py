#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMD 膜数据 XGBoost + SHAP 可解释性分析完整脚本
============================================================
输入:  cleaned_OMD_data.csv (540 行 x 41 列, 已完成初步清洗)
输出:
    shap_summary_logA.png            log_A 全局 SHAP 汇总图
    shap_dependence_logA_top1.png    log_A 排名第 1 特征的依赖图
    shap_summary_Robs.png            Robs 全局 SHAP 汇总图
    shap_dependence_hydraulicP_bar.png
    shap_dependence_NaCl_M.png
    shap_dependence_CP.png
    pdp_2D_Pressure_NaCl.png         压力 x 盐浓度 二维偏依赖图
    SI_Data_Distribution.png         SI 采样分布直方图

运行:  python omd_xgb_shap_analysis.py
依赖:  pandas numpy matplotlib seaborn xgboost scikit-learn shap
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import KFold, cross_val_score

# ============================================================
# 全局配置
# ============================================================
DATA_PATH = "cleaned_OMD_data.csv"
SEED = 42
DPI = 300

# 模型超参 (两次建模保持一致)
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
)

# 建模时剔除的列: 标识符 + 目标变量及其派生量
DROP_COLS = ["id", "name", "reportReference",
             "A", "B", "log_A", "log_B", "Robs"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.bbox": "tight",
})


def banner(msg):
    print("\n" + "=" * 62)
    print(msg)
    print("=" * 62)


# ============================================================
# 步骤 1: 读取数据 + Robs 小数→百分数修正
# ============================================================
banner("步骤 1 | 读取数据与 Robs 修正")

df = pd.read_csv(DATA_PATH)
print(f"原始形状: {df.shape}")

# OMD 偶尔把 Robs 导成小数形式 (如 0.869 应为 86.9 %)
n_fixed = int((df["Robs"] <= 1).sum())
df.loc[df["Robs"] <= 1, "Robs"] = df.loc[df["Robs"] <= 1, "Robs"] * 100
print(f"Robs <= 1 的样本数: {n_fixed}  → 已乘以 100")
print(f"修正后 Robs 范围: {df['Robs'].min():.2f} ~ {df['Robs'].max():.2f}")

# ============================================================
# 步骤 2: 构建特征矩阵
# ============================================================
banner("步骤 2 | 构建特征矩阵")

X = df.drop(columns=DROP_COLS)
y_logA = df["log_A"]
y_robs = df["Robs"]

print(f"特征数: {X.shape[1]}")
print(f"样本数: {X.shape[0]}")
print(f"缺失值总数: {int(X.isna().sum().sum())}")
print(f"log_A: mean={y_logA.mean():.3f}, std={y_logA.std():.3f}")
print(f"Robs : mean={y_robs.mean():.2f}, std={y_robs.std():.2f}")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)


def cv_report(model, y, label):
    """5 折交叉验证并打印 R2 / RMSE"""
    r2 = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    rmse = -cross_val_score(model, X, y, cv=kf,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"\n[{label}]")
    print(f"  每折 R2  : {np.round(r2, 4).tolist()}")
    print(f"  R2       : {r2.mean():.4f} ± {r2.std():.4f}")
    print(f"  RMSE     : {rmse.mean():.3f} ± {rmse.std():.3f}")
    return r2, rmse


# ============================================================
# 步骤 3: log_A 模型 (水渗透率) + SHAP
# ============================================================
banner("步骤 3 | log_A 模型训练与 SHAP 分析")

model_logA = xgb.XGBRegressor(**XGB_PARAMS)
cv_report(model_logA, y_logA, "XGBoost — log_A")
model_logA.fit(X, y_logA)

explainer_logA = shap.TreeExplainer(model_logA)
sv_logA = explainer_logA.shap_values(X)
if isinstance(sv_logA, list):          # 兼容旧版 shap 的多输出返回
    sv_logA = sv_logA[0]

# --- 3.1 全局汇总图 ---
plt.figure(figsize=(10, 8))
shap.summary_plot(sv_logA, X, max_display=20, show=False)
plt.tight_layout()
plt.savefig("shap_summary_logA.png", dpi=DPI)
plt.close()

# --- 3.2 排名第 1 特征的依赖图 ---
mean_abs_logA = np.abs(sv_logA).mean(axis=0)
top_feat_logA = X.columns[int(np.argmax(mean_abs_logA))]
print(f"\n排名第 1 特征: {top_feat_logA}  "
      f"(mean|SHAP| = {mean_abs_logA.max():.4f})")

plt.figure(figsize=(8, 6))
shap.dependence_plot(top_feat_logA, sv_logA, X, show=False)
plt.tight_layout()
plt.savefig("shap_dependence_logA_top1.png", dpi=DPI)
plt.close()

# ============================================================
# 步骤 4: Robs 模型 (盐截留率) + SHAP + 2D PDP
# ============================================================
banner("步骤 4 | Robs 模型训练与 SHAP 分析")

model_robs = xgb.XGBRegressor(**XGB_PARAMS)
cv_report(model_robs, y_robs, "XGBoost — Robs")

rf = RandomForestRegressor(n_estimators=500, max_depth=8,
                           random_state=SEED, n_jobs=-1)
cv_report(rf, y_robs, "RandomForest 基线 — Robs")

model_robs.fit(X, y_robs)

explainer_robs = shap.TreeExplainer(model_robs)
sv_robs = explainer_robs.shap_values(X)
if isinstance(sv_robs, list):
    sv_robs = sv_robs[0]

# --- 4.1 全局汇总图 ---
plt.figure(figsize=(10, 8))
shap.summary_plot(sv_robs, X, max_display=20, show=False)
plt.tight_layout()
plt.savefig("shap_summary_Robs.png", dpi=DPI)
plt.close()

# --- 4.2 打印 mean|SHAP| 排名 ---
mean_abs_robs = pd.Series(np.abs(sv_robs).mean(axis=0), index=X.columns)
print("\nmean|SHAP| Top 10:")
print(mean_abs_robs.sort_values(ascending=False).head(10).round(4).to_string())

# 操作条件 + 流体力学 vs 膜化学 贡献占比
op_keys = ["hydraulicP_bar", "CP", "NaCl_M", "CP_is_missing",
           "filtrationMode_Cross-flow", "filtrationMode_Dead-end",
           "filtrationMode_Not available",
           "cpMode_As reported", "cpMode_CP calculated assuming kf = 100",
           "cpMode_Not available"]
chem_keys = [c for c in X.columns
             if c.startswith(("chemistry__", "structure__", "modification__"))]
op_share = mean_abs_robs[[c for c in op_keys if c in X.columns]].sum()
chem_share = mean_abs_robs[chem_keys].sum()
total = mean_abs_robs.sum()
print(f"\n操作+流体力学: {op_share / total * 100:.1f} %")
print(f"膜化学      : {chem_share / total * 100:.1f} %")

# --- 4.3 三个关键特征的依赖图 ---
for feat in ["hydraulicP_bar", "NaCl_M", "CP"]:
    if feat not in X.columns:
        print(f"[跳过] 特征 {feat} 不存在")
        continue
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(feat, sv_robs, X, show=False)
    plt.tight_layout()
    plt.savefig(f"shap_dependence_{feat}.png", dpi=DPI)
    plt.close()
    print(f"已保存: shap_dependence_{feat}.png")

# --- 4.4 二维偏依赖图 (压力 x 盐浓度) ---
fig, ax = plt.subplots(figsize=(10, 8))
PartialDependenceDisplay.from_estimator(
    model_robs, X,
    features=[("hydraulicP_bar", "NaCl_M")],
    grid_resolution=40,
    kind="average",
    ax=ax,
)
plt.tight_layout()
plt.savefig("pdp_2D_Pressure_NaCl.png", dpi=DPI)
plt.close()
print("已保存: pdp_2D_Pressure_NaCl.png")

# ============================================================
# 步骤 5: SI 采样分布直方图
# ============================================================
banner("步骤 5 | SI 数据分布直方图")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, col, color, xlabel, tag in zip(
    axes,
    ["hydraulicP_bar", "NaCl_M"],
    ["#2166AC", "#4DAF4A"],
    ["Hydraulic Pressure (bar)", "NaCl Concentration (M)"],
    ["(a)", "(b)"],
):
    x = df[col].dropna()
    bins = 25 if col == "hydraulicP_bar" else 30
    sns.histplot(x, bins=bins, kde=True, ax=ax,
                 color=color, edgecolor="white", linewidth=0.4,
                 stat="count", alpha=0.9)
    ax.axvline(x.median(), color="#B2182B", linestyle="--", linewidth=1.6,
               label=f"Median = {x.median():.3g}")
    ax.axvline(x.mean(), color="#2C7BB6", linestyle=":", linewidth=1.6,
               label=f"Mean   = {x.mean():.3g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sample Count")
    ax.set_title(tag, loc="left", fontweight="bold", pad=4)
    ax.legend(frameon=False, fontsize=10)
    sns.despine(ax=ax, top=True, right=True, trim=False)
    print(f"{col}: n={len(x)}, median={x.median():.4g}, mean={x.mean():.4g}")

plt.tight_layout(w_pad=2.5)
plt.savefig("SI_Data_Distribution.png", dpi=DPI)
plt.close()
print("已保存: SI_Data_Distribution.png")

# ============================================================
# 完成
# ============================================================
banner("全部完成")
print("生成文件清单:")
for f in ["shap_summary_logA.png", "shap_dependence_logA_top1.png",
          "shap_summary_Robs.png", "shap_dependence_hydraulicP_bar.png",
          "shap_dependence_NaCl_M.png", "shap_dependence_CP.png",
          "pdp_2D_Pressure_NaCl.png", "SI_Data_Distribution.png"]:
    print(f"  - {f}")
