"""
Phase 7: Exploratory Data Analysis (EDA) Module
EMIPredict AI - Intelligent Financial Risk Assessment Platform

Generates publication-quality figures and exact statistical summaries
using strictly data/processed/train.csv (keeping test.csv untouched).
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Style configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300
})

TRAIN_PATH = Path("data/processed/train.csv")
FIGURES_DIR = Path("reports/figures")


def generate_eda():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading training dataset from {TRAIN_PATH}...")
    df = pd.read_csv(TRAIN_PATH, low_memory=False)
    n_rows = len(df)
    print(f"Loaded {n_rows:,} rows across {len(df.columns)} columns.\n")

    stats = {}

    # ---------------------------------------------------------
    # 1. Target Distributions
    # ---------------------------------------------------------
    print("Generating Figure 1: Target Distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1a. Classification Target
    cls_counts = df["emi_eligibility"].value_counts()
    cls_pcts = df["emi_eligibility"].value_counts(normalize=True) * 100
    stats["cls_counts"] = cls_counts.to_dict()
    stats["cls_pcts"] = cls_pcts.to_dict()

    palette_cls = {"Not_Eligible": "#d9534f", "Eligible": "#5cb85c", "High_Risk": "#f0ad4e"}
    bars = sns.barplot(
        x=cls_counts.index,
        y=cls_counts.values,
        ax=axes[0],
        palette=palette_cls
    )
    axes[0].set_title("A: Classification Target (emi_eligibility)")
    axes[0].set_xlabel("Eligibility Category")
    axes[0].set_ylabel("Applicant Count")
    for bar in bars.patches:
        h = bar.get_height()
        pct = (h / n_rows) * 100
        axes[0].annotate(
            f"{int(h):,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10
        )

    # 1b. Regression Target Histogram + KDE
    target_emi = df["max_monthly_emi"]
    stats["emi_floor_count"] = int((target_emi == 500).sum())
    stats["emi_floor_pct"] = float((target_emi == 500).mean() * 100)
    stats["emi_gt_50k_count"] = int((target_emi > 50000).sum())
    stats["emi_gt_50k_pct"] = float((target_emi > 50000).mean() * 100)

    # Clip display to 40k for main histogram to show distribution clearly
    sns.histplot(
        target_emi[target_emi <= 40000],
        bins=50,
        kde=True,
        ax=axes[1],
        color="#337ab7"
    )
    axes[1].axvline(500, color="#d9534f", linestyle="--", linewidth=2, label="₹500 Floor (26.4%)")
    axes[1].axvline(target_emi.median(), color="#5cb85c", linestyle="-", linewidth=2, label=f"Median: ₹{target_emi.median():,.0f}")
    axes[1].set_title("B: Regression Target (max_monthly_emi ≤ ₹40,000)")
    axes[1].set_xlabel("Max Monthly EMI (₹)")
    axes[1].set_ylabel("Count")
    axes[1].legend(loc="upper right")

    # 1c. Boxplot by Class
    sns.boxplot(
        data=df,
        x="emi_eligibility",
        y="max_monthly_emi",
        ax=axes[2],
        palette=palette_cls,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    axes[2].set_title("C: Max Monthly EMI by Eligibility Class")
    axes[2].set_xlabel("Eligibility Category")
    axes[2].set_ylabel("Max Monthly EMI (₹)")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Max Monthly EMI (₹, Log Scale)")

    plt.suptitle("Target Distributions & Multi-Class Structure (Training Set, N=323,840)", y=1.02)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "01_target_distributions.png"
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # 2. Missing Values Profile
    # ---------------------------------------------------------
    print("Generating Figure 2: Missing Values Profile...")
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)
    missing_pcts = (missing_counts / n_rows) * 100
    stats["missing"] = {col: {"count": int(c), "pct": float(p)} for col, c, p in zip(missing_counts.index, missing_counts.values, missing_pcts.values)}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(missing_counts.index, missing_pcts.values, color="#5bc0de", edgecolor="#269abc")
    ax.set_title("Missing Values Percentage by Feature (Training Set, N=323,840)")
    ax.set_xlabel("Missing Percentage (%)")
    ax.set_ylabel("Feature Name")
    ax.set_xlim(0, 1.2)
    for bar, count in zip(bars, missing_counts.values):
        w = bar.get_width()
        ax.annotate(
            f"{w:.2f}% ({count:,} rows)",
            xy=(w + 0.02, bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            fontsize=10,
            color="#333333"
        )
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "02_missing_values_profile.png"
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # 3. Credit Score Analysis & Anomalies
    # ---------------------------------------------------------
    print("Generating Figure 3: Credit Score Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    cs = df["credit_score"].dropna()
    sns.histplot(cs, bins=60, kde=False, ax=axes[0], color="#204060")
    axes[0].axvspan(0, 0.5, color="#d9534f", alpha=0.3, label=f"Score = 0: NTC {(df['credit_score']==0).sum():,} rows (0.4%)")
    axes[0].axvspan(300, 850, color="#5cb85c", alpha=0.15, label="Standard CIBIL/FICO (300-850)")
    axes[0].axvspan(850, 1200, color="#f0ad4e", alpha=0.3, label=f"Score > 850: Synthetic Tail {(df['credit_score']>850).sum():,} rows (0.8%)")
    axes[0].set_title("A: Credit Score Distribution with Anomaly Regions")
    axes[0].set_xlabel("Credit Score")
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="upper right")

    # Credit score tiers vs Eligibility
    bins = [-np.inf, -0.1, 0.5, 599, 749, 850, np.inf]
    labels = ["Negative", "Zero (NTC)", "300-599 (Poor)", "600-749 (Fair/Good)", "750-850 (Excellent)", ">850 (Synthetic)"]
    df_cs_tier = pd.cut(df["credit_score"], bins=bins, labels=labels)
    ct = pd.crosstab(df_cs_tier, df["emi_eligibility"], normalize="index") * 100
    stats["cs_elig_crosstab"] = ct.round(2).to_dict()

    ct.plot(
        kind="bar",
        stacked=True,
        color=["#5cb85c", "#f0ad4e", "#d9534f"],
        ax=axes[1],
        edgecolor="black",
        linewidth=0.5
    )
    axes[1].set_title("B: Eligibility Breakdown Across Credit Score Tiers")
    axes[1].set_xlabel("Credit Score Tier")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend(title="Eligibility", loc="upper right")
    plt.xticks(rotation=25, ha="right")

    plt.suptitle("Credit Score Anatomy & Synthetic Distribution Verification", y=1.02)
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "03_credit_score_analysis.png"
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # 4. Financial Capacity, Installment Ratio & FOIR
    # ---------------------------------------------------------
    print("Generating Figure 4: Financial Capacity & Affordability Ratios...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Calculate affordability ratio and projected FOIR
    req_inst = df["requested_amount"] / df["requested_tenure"]
    inst_to_max = req_inst / df["max_monthly_emi"]
    proj_foir = (df["current_emi_amount"] + req_inst) / df["monthly_salary"]
    df_plot = df.assign(inst_to_max=inst_to_max, proj_foir=proj_foir)

    # 4a. Installment to Max EMI Ratio (log scale)
    sns.boxplot(
        data=df_plot,
        x="emi_eligibility",
        y="inst_to_max",
        ax=axes[0],
        palette=palette_cls,
        showfliers=False
    )
    axes[0].axhline(1.0, color="#d9534f", linestyle="--", linewidth=1.5, label="Affordability Boundary (Ratio = 1.0)")
    axes[0].set_title("A: Requested Installment / Max Monthly EMI")
    axes[0].set_xlabel("Eligibility Category")
    axes[0].set_ylabel("Installment to Max EMI Ratio")
    axes[0].legend(loc="upper left")

    # 4b. Projected FOIR
    sns.boxplot(
        data=df_plot,
        x="emi_eligibility",
        y="proj_foir",
        ax=axes[1],
        palette=palette_cls,
        showfliers=False
    )
    axes[1].axhline(0.50, color="#f0ad4e", linestyle=":", linewidth=1.5, label="Standard 50% FOIR Benchmark")
    axes[1].set_title("B: Projected Total Obligations to Salary (FOIR)")
    axes[1].set_xlabel("Eligibility Category")
    axes[1].set_ylabel("Projected FOIR Ratio")
    axes[1].legend(loc="upper left")

    # 4c. Monthly Salary vs Max Monthly EMI
    sample_sub = df_plot.sample(n=3000, random_state=42)
    sns.scatterplot(
        data=sample_sub,
        x="monthly_salary",
        y="max_monthly_emi",
        hue="emi_eligibility",
        palette=palette_cls,
        alpha=0.6,
        s=20,
        ax=axes[2]
    )
    axes[2].set_title("C: Monthly Salary vs Max EMI (Sample n=3,000)")
    axes[2].set_xlabel("Monthly Salary (₹)")
    axes[2].set_ylabel("Max Monthly EMI (₹)")
    axes[2].set_xlim(0, 250000)
    axes[2].set_ylim(0, 60000)

    plt.suptitle("Debt Capacity, Obligation Ratios & Latent Affordability", y=1.02)
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "04_financial_capacity_and_foir.png"
    plt.savefig(fig4_path, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # 5. Comparisons Across EMI Scenarios
    # ---------------------------------------------------------
    print("Generating Figure 5: Scenario Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Scenario vs Eligibility
    scen_ct = pd.crosstab(df["emi_scenario"], df["emi_eligibility"], normalize="index") * 100
    stats["scen_ct"] = scen_ct.round(2).to_dict()

    scen_ct.plot(
        kind="barh",
        stacked=True,
        color=["#5cb85c", "#f0ad4e", "#d9534f"],
        ax=axes[0],
        edgecolor="black",
        linewidth=0.5
    )
    axes[0].set_title("A: Eligibility Proportions Across EMI Scenarios")
    axes[0].set_xlabel("Percentage (%)")
    axes[0].set_ylabel("EMI Scenario")
    axes[0].legend(title="Eligibility", loc="lower right")

    # Scenario vs Requested Amount Boxplot
    sns.boxplot(
        data=df,
        y="emi_scenario",
        x="requested_amount",
        ax=axes[1],
        palette="Blues_r",
        showfliers=False
    )
    axes[1].set_title("B: Requested Loan Amount Distribution by Scenario")
    axes[1].set_xlabel("Requested Loan Amount (₹)")
    axes[1].set_ylabel("")

    plt.suptitle("EMI Scenario Segmentation & Product Ticket-Size Profiles", y=1.02)
    plt.tight_layout()
    fig5_path = FIGURES_DIR / "05_scenario_comparison.png"
    plt.savefig(fig5_path, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # 6. Correlation Heatmap
    # ---------------------------------------------------------
    print("Generating Figure 6: Correlation Heatmap...")
    key_num_cols = [
        "monthly_salary", "monthly_rent", "groceries_utilities",
        "travel_expenses", "other_monthly_expenses", "current_emi_amount",
        "credit_score", "bank_balance", "emergency_fund",
        "requested_amount", "requested_tenure", "max_monthly_emi"
    ]
    corr_mat = df[key_num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_mat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.35,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Pearson Correlation Heatmap of Key Financial & Loan Features")
    plt.tight_layout()
    fig6_path = FIGURES_DIR / "06_correlation_heatmap.png"
    plt.savefig(fig6_path, bbox_inches="tight")
    plt.close()

    print("All 6 figures successfully saved in reports/figures/.")
    return stats


if __name__ == "__main__":
    generate_eda()
