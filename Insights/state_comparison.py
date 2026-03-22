"""
state_comparison.py — generates state-level charts from preprocessed DuckDB tables
Run: python state_comparison.py --db ../database/aadhar.duckdb
Outputs: state_output/
"""
import argparse, warnings
from pathlib import Path
import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
OUTPUT_DIR = Path("state_output")
OUTPUT_DIR.mkdir(exist_ok=True)

MONTHLY_DATES = ['2025-04-01','2025-05-01','2025-06-01','2025-07-01']
TEAL="#1D9E75"; PURPLE="#534AB7"; AMBER="#BA7517"; CORAL="#D85A30"; GRAY="#9ca3af"; BG="#f9f8f6"

def setup_style():
    plt.rcParams.update({
        "figure.facecolor":BG,"axes.facecolor":"#ffffff",
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.spines.left":False,"axes.grid":True,
        "axes.grid.axis":"x","grid.color":"#f0f0f0","grid.linewidth":0.8,
        "font.family":"DejaVu Sans","font.size":11,
        "xtick.labelsize":9,"ytick.labelsize":10,
    })

def fetch_state_data(db_path):
    con = duckdb.connect(db_path, read_only=True)
    dts = ", ".join([f"'{d}'" for d in MONTHLY_DATES])
    bio = con.execute(f"""
        SELECT state,
            SUM(bio_total) AS bio_total,
            AVG(age_5_ratio) AS avg_age_5_ratio,
            AVG(age_17_ratio) AS avg_age_17_ratio,
            AVG(dependency_ratio) AS avg_dependency_ratio,
            AVG(daily_pct_change) AS avg_bio_growth,
            AVG(bio_total_7day_std) AS avg_bio_volatility,
            COUNT(DISTINCT district) AS district_count
        FROM biometric_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts}) AND state IS NOT NULL
        GROUP BY state""").fetchdf()
    demo = con.execute(f"""
        SELECT state,
            SUM(demo_total) AS demo_total,
            AVG(demo_age_5_ratio) AS avg_demo_5_ratio,
            AVG(demo_dependency_ratio) AS avg_demo_dependency,
            AVG(daily_pct_change) AS avg_demo_growth
        FROM demographic_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts}) AND state IS NOT NULL
        GROUP BY state""").fetchdf()
    enrol = con.execute(f"""
        SELECT state,
            SUM(enrol_total) AS enrol_total,
            AVG(enrol_minor_ratio) AS avg_minor_ratio,
            AVG(enrol_adult_ratio) AS avg_adult_ratio,
            AVG(daily_pct_change) AS avg_enrol_growth,
            AVG(enrol_total_7day_std) AS avg_enrol_volatility
        FROM enrolment_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts}) AND state IS NOT NULL
        GROUP BY state""").fetchdf()
    monthly = con.execute(f"""
        SELECT state, CAST(date AS DATE)::VARCHAR AS month,
            AVG(dependency_ratio) AS dependency_ratio,
            SUM(bio_total) AS bio_total
        FROM biometric_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts}) AND state IS NOT NULL
        GROUP BY state, date ORDER BY state, date""").fetchdf()
    con.close()
    df = bio.merge(demo, on="state", how="outer").merge(enrol, on="state", how="outer").fillna(0)
    df['state'] = df['state'].str.strip()
    df = df[df['state'] != ""].copy()
    print(f"  States fetched: {len(df)}")
    return df, monthly

def fmt(x, _):
    return f"{x/1e6:.1f}M" if x>=1e6 else f"{x/1e3:.0f}K"

def plot_enrolment_bar(df):
    ranked = df.sort_values("enrol_total", ascending=True).tail(30)
    fig, ax = plt.subplots(figsize=(10,11), facecolor=BG); ax.set_facecolor("#ffffff")
    bars = ax.barh(ranked['state'], ranked['enrol_total'], color=TEAL, alpha=0.85, height=0.65)
    top5 = ranked['enrol_total'].nlargest(5).index
    for bar, idx in zip(bars, ranked.index):
        if idx in top5: bar.set_color(PURPLE); bar.set_alpha(0.9)
    ax.set_xlabel("Total enrolment (Apr–Jul 2025)", fontsize=10)
    ax.set_title("State-level Aadhaar enrolment — ranked", fontsize=14, fontweight='bold', pad=14, loc='left')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt))
    mx = ranked['enrol_total'].max()
    for bar in bars:
        w = bar.get_width()
        lbl = f"{w/1e6:.2f}M" if w>=1e6 else f"{w/1e3:.0f}K"
        ax.text(w + mx*0.01, bar.get_y()+bar.get_height()/2, lbl, va='center', fontsize=8, color='#4b5563')
    ax.spines['bottom'].set_color('#e5e7eb')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"state_enrolment_bar.png", dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(); print("  Saved → state_enrolment_bar.png")

def plot_adult_vs_minor(df):
    ranked = df.sort_values("avg_adult_ratio", ascending=True)
    fig, ax = plt.subplots(figsize=(10,11), facecolor=BG); ax.set_facecolor("#ffffff")
    y = range(len(ranked))
    ax.barh(y, ranked['avg_minor_ratio'], color=TEAL, alpha=0.85, height=0.65, label="Minor (0–17)")
    ax.barh(y, ranked['avg_adult_ratio'], color=PURPLE, alpha=0.85, height=0.65, left=ranked['avg_minor_ratio'], label="Adult (18+)")
    ax.set_yticks(list(y)); ax.set_yticklabels(ranked['state'], fontsize=9)
    ax.set_xlabel("Proportion of total enrolment", fontsize=10)
    ax.set_title("Adult vs minor enrolment ratio by state", fontsize=14, fontweight='bold', pad=14, loc='left')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_xlim(0,1); ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axvline(0.5, color='#d1d5db', linewidth=1, linestyle='--', alpha=0.7)
    ax.spines['bottom'].set_color('#e5e7eb')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"state_adult_vs_minor.png", dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(); print("  Saved → state_adult_vs_minor.png")

def plot_growth_bar(df):
    ranked = df.sort_values("avg_enrol_growth", ascending=True).tail(30)
    fig, ax = plt.subplots(figsize=(10,11), facecolor=BG); ax.set_facecolor("#ffffff")
    med = ranked['avg_enrol_growth'].median()
    colors = [CORAL if v > med else GRAY for v in ranked['avg_enrol_growth']]
    ax.barh(ranked['state'], ranked['avg_enrol_growth'], color=colors, alpha=0.85, height=0.65)
    ax.set_xlabel("Avg daily % change in enrolment", fontsize=10)
    ax.set_title("Enrolment growth momentum by state", fontsize=14, fontweight='bold', pad=14, loc='left')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}%"))
    ax.axvline(med, color=PURPLE, linewidth=1.2, linestyle='--', alpha=0.8)
    ax.text(med, len(ranked)-0.5, f" median\n {med:.2f}%", fontsize=8, color=PURPLE, va='top')
    ax.spines['bottom'].set_color('#e5e7eb')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"state_growth_bar.png", dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(); print("  Saved → state_growth_bar.png")

def plot_dependency_heatmap(monthly):
    pivot = monthly.pivot_table(index='state', columns='month', values='dependency_ratio', aggfunc='mean')
    pivot = pivot.dropna(how='all')
    col_map = {'2025-04-01':'Apr','2025-05-01':'May','2025-06-01':'Jun','2025-07-01':'Jul'}
    pivot = pivot.rename(columns=col_map)
    pivot = pivot[[c for c in ['Apr','May','Jun','Jul'] if c in pivot.columns]]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index].head(35)
    fig, ax = plt.subplots(figsize=(8,12), facecolor=BG)
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.4, linecolor='#f0f0f0',
                annot=True, fmt=".2f", annot_kws={"size":8},
                cbar_kws={"shrink":0.5,"label":"Dependency ratio"})
    ax.set_title("Child-to-adult dependency ratio — state × month", fontsize=13, fontweight='bold', pad=14, loc='left')
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=9); ax.tick_params(axis='x', labelsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"state_dependency_heatmap.png", dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(); print("  Saved → state_dependency_heatmap.png")

def plot_district_scatter(df):
    fig, ax = plt.subplots(figsize=(11,7), facecolor=BG); ax.set_facecolor("#ffffff")
    sizes = (df['enrol_total'] / df['enrol_total'].max()) * 1200 + 30
    sc = ax.scatter(df['district_count'], df['avg_enrol_growth'], s=sizes,
                    c=df['avg_adult_ratio'], cmap='RdYlGn', alpha=0.75, linewidths=0.5, edgecolors='white')
    plt.colorbar(sc, ax=ax, shrink=0.6).set_label("Adult enrolment ratio", fontsize=9)
    for _, row in df.nlargest(12,'enrol_total').iterrows():
        ax.annotate(row['state'][:14], (row['district_count'], row['avg_enrol_growth']),
                    xytext=(6,4), textcoords='offset points', fontsize=7.5, color='#374151')
    ax.set_xlabel("Number of districts", fontsize=10)
    ax.set_ylabel("Avg daily enrolment growth %", fontsize=10)
    ax.set_title("State size vs enrolment growth\nBubble size = total enrolment · Colour = adult ratio",
                 fontsize=13, fontweight='bold', pad=12, loc='left')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color('#e5e7eb')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"state_scatter.png", dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(); print("  Saved → state_scatter.png")

def save_summary(df):
    out = df.sort_values('enrol_total', ascending=False)
    out.insert(0,'enrol_rank', range(1, len(out)+1))
    out.to_csv(OUTPUT_DIR/"state_summary.csv", index=False)
    print(f"  Saved → state_summary.csv  ({len(out)} states)")

def main(db_path):
    print("="*60); print("STATE-LEVEL COMPARISON"); print("="*60)
    setup_style()
    print("\n── Fetching ──"); df, monthly = fetch_state_data(db_path)
    print("\n── Charts ──")
    plot_enrolment_bar(df); plot_adult_vs_minor(df); plot_growth_bar(df)
    plot_dependency_heatmap(monthly); plot_district_scatter(df); save_summary(df)
    print("\nDONE — state_output/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    main(p.parse_args().db)