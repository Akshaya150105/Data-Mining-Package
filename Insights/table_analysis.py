"""
table_analysis.py
==================
Complete analysis of all 3 Aadhaar tables — Biometric, Enrolment, Demographic.
No district filter. Covers all records across all dates.

Outputs (in table_output/):

  BIOMETRIC
    bio_age_distribution.png       age group totals across all India
    bio_state_heatmap.png          state x month dependency heatmap
    bio_top_bottom.png             top 10 vs bottom 10 states
    bio_growth_trend.png           national daily trend + 7-day avg
    bio_volatility_map.png         state volatility ranking
    bio_ratio_scatter.png          age5 ratio vs dependency scatter

  ENROLMENT
    enrol_age_pie.png              national age group pie
    enrol_adult_minor_bar.png      adult vs minor stacked by state
    enrol_growth_heatmap.png       state x month growth heatmap
    enrol_top_growth.png           fastest growing states
    enrol_volatility.png           most volatile states
    enrol_trend.png                national daily trend

  DEMOGRAPHIC
    demo_dependency_dist.png       dependency ratio distribution
    demo_state_comparison.png      state avg demo totals
    demo_age_ratio_heatmap.png     state x month age5 ratio heatmap
    demo_trend.png                 national daily trend
    demo_vs_bio_scatter.png        demo dependency vs bio dependency

  COMBINED
    all_tables_trend.png           bio + enrol + demo on same chart
    correlation_heatmap.png        feature correlation matrix

Usage:
    python table_analysis.py --db ../database/aadhar.duckdb
"""

import argparse, warnings
from pathlib import Path
import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
OUTPUT_DIR = Path("table_output")
OUTPUT_DIR.mkdir(exist_ok=True)

BG     = "#f9f8f6"
WHITE  = "#ffffff"
C_BIO  = "#534AB7"
C_EN   = "#1D9E75"
C_DEM  = "#BA7517"
C_NEG  = "#D85A30"
C_ALT  = "#9FE1CB"
C_ALT2 = "#FAC775"
C_ALT3 = "#CECBF6"

def setup():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": WHITE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#f0f0f0", "grid.linewidth": 0.7,
        "font.family": "DejaVu Sans", "font.size": 11,
    })

def save(fig, name):
    fig.savefig(OUTPUT_DIR / name, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {name}")

def fmt_y(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x,_: f"{x/1e6:.1f}M" if x>=1e6 else f"{x/1e3:.0f}K" if x>=1e3 else f"{x:.2f}"))

def fmt_x(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x,_: f"{x/1e6:.1f}M" if x>=1e6 else f"{x/1e3:.0f}K" if x>=1e3 else f"{x:.2f}"))

def month_lines(ax, ymax):
    for ms in ["2025-09-01","2025-10-01","2025-11-01","2025-12-01"]:
        ax.axvline(pd.Timestamp(ms), color="#e5e7eb", linewidth=1, zorder=1)
        ax.text(pd.Timestamp(ms), ymax*1.01,
                f" {pd.Timestamp(ms).strftime('%b')}", fontsize=8, color="#9ca3af")

def ax_style(ax):
    ax.spines["bottom"].set_color("#e5e7eb")
    ax.spines["left"].set_color("#e5e7eb")


# ══════════════════════════════════════════════════════════════════════════
# FETCH  — complete tables, no district filter
# ══════════════════════════════════════════════════════════════════════════

def fetch(db_path):
    con = duckdb.connect(db_path, read_only=True)

    bio_d   = set(con.execute("SELECT DISTINCT CAST(date AS DATE)::VARCHAR FROM biometric_data_preprocessed").df().iloc[:,0])
    demo_d  = set(con.execute("SELECT DISTINCT CAST(date AS DATE)::VARCHAR FROM demographic_data_preprocessed").df().iloc[:,0])
    enrol_d = set(con.execute("SELECT DISTINCT CAST(date AS DATE)::VARCHAR FROM enrolment_data_preprocessed").df().iloc[:,0])
    all_dates = sorted(bio_d & demo_d & enrol_d)
    dts = ", ".join([f"'{d}'" for d in all_dates])
    print(f"  Dates: {len(all_dates)}  ({all_dates[0]} → {all_dates[-1]})")

    # ── Biometric: aggregate per state per date ────────────────────────
    bio = con.execute(f"""
        SELECT
            CAST(date AS DATE)::VARCHAR AS date,
            state,
            SUM(bio_age_5_17)           AS bio_age_5_17,
            SUM(bio_age_17_)            AS bio_age_17_plus,
            SUM(bio_total)              AS bio_total,
            AVG(age_5_ratio)            AS age_5_ratio,
            AVG(age_17_ratio)           AS age_17_ratio,
            AVG(dependency_ratio)       AS dependency_ratio,
            AVG(bio_total_7day_avg)     AS bio_7day_avg,
            AVG(bio_total_7day_std)     AS bio_7day_std,
            AVG(daily_pct_change)       AS bio_growth_pct,
            AVG(daily_change)           AS bio_daily_change
        FROM biometric_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts}) AND state IS NOT NULL
        GROUP BY date, state ORDER BY date, state
    """).fetchdf()

    # ── Enrolment: aggregate per state per date ────────────────────────
    enrol = con.execute(f"""
        SELECT
            CAST(date AS DATE)::VARCHAR AS date,
            state,
            SUM(age_0_5)                AS age_0_5,
            SUM(age_5_17)               AS age_5_17,
            SUM(age_18_greater)         AS age_18_plus,
            SUM(enrol_total)            AS enrol_total,
            AVG(enrol_minor_ratio)      AS minor_ratio,
            AVG(enrol_adult_ratio)      AS adult_ratio,
            AVG(enrol_total_7day_avg)   AS enrol_7day_avg,
            AVG(enrol_total_7day_std)   AS enrol_7day_std,
            AVG(daily_pct_change)       AS enrol_growth_pct,
            AVG(daily_change)           AS enrol_daily_change
        FROM enrolment_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts}) AND state IS NOT NULL
        GROUP BY date, state ORDER BY date, state
    """).fetchdf()

    # ── Demographic: aggregate per state per date ──────────────────────
    demo = con.execute(f"""
        SELECT
            CAST(date AS DATE)::VARCHAR AS date,
            state,
            SUM(demo_age_5_17)          AS demo_age_5_17,
            SUM(demo_age_17_)           AS demo_age_17_plus,
            SUM(demo_total)             AS demo_total,
            AVG(demo_age_5_ratio)       AS demo_age5_ratio,
            AVG(demo_age_17_ratio)      AS demo_age17_ratio,
            AVG(demo_dependency_ratio)  AS demo_dependency,
            AVG(demo_total_7day_avg)    AS demo_7day_avg,
            AVG(demo_total_7day_std)    AS demo_7day_std,
            AVG(daily_pct_change)       AS demo_growth_pct
        FROM demographic_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts}) AND state IS NOT NULL
        GROUP BY date, state ORDER BY date, state
    """).fetchdf()

    con.close()

    for df in [bio, enrol, demo]:
        df["date"]  = pd.to_datetime(df["date"])
        df["state"] = df["state"].str.strip()
        df["month"] = df["date"].dt.to_period("M").astype(str)

    print(f"  Bio rows: {len(bio):,}  |  Enrol: {len(enrol):,}  |  Demo: {len(demo):,}")
    return bio, enrol, demo


# ══════════════════════════════════════════════════════════════════════════
# BIOMETRIC CHARTS
# ══════════════════════════════════════════════════════════════════════════

def bio_charts(bio):
    print("\n── Biometric charts ──")

    # 1. National age group totals bar
    age_totals = {
        "Age 5–17": bio["bio_age_5_17"].sum(),
        "Age 17+":  bio["bio_age_17_plus"].sum(),
    }
    fig, ax = plt.subplots(figsize=(7,4), facecolor=BG)
    bars = ax.bar(age_totals.keys(), age_totals.values(),
                  color=[C_BIO, C_ALT3], alpha=0.85, width=0.4)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()*1.01,
                f"{bar.get_height()/1e6:.2f}M",
                ha="center", fontsize=10, color="#374151")
    fmt_y(ax)
    ax.set_title("National biometric enrolment by age group",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    ax_style(ax)
    save(fig, "bio_age_distribution.png")

    # 2. State x month dependency heatmap
    pivot = bio.pivot_table(index="state", columns="month",
                            values="dependency_ratio", aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index].head(30)
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
                linecolor="#f0f0f0", annot=True, fmt=".2f",
                annot_kws={"size":7},
                cbar_kws={"shrink":0.5, "label":"Dependency ratio"})
    ax.set_title("Biometric dependency ratio — state × month",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8); ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    save(fig, "bio_state_heatmap.png")

    # 3. Top 10 vs Bottom 10 states by bio total
    state_tot = bio.groupby("state")["bio_total"].sum().sort_values(ascending=False)
    top10    = state_tot.head(10)
    bottom10 = state_tot.tail(10)
    fig, axes = plt.subplots(1,2, figsize=(13,5), facecolor=BG)
    for ax, data, color, title in [
        (axes[0], top10,    C_BIO,  "Top 10 states — bio total"),
        (axes[1], bottom10, C_NEG,  "Bottom 10 states — bio total"),
    ]:
        ax.set_facecolor(WHITE)
        ax.barh(data.index, data.values, color=color, alpha=0.85, height=0.6)
        fmt_x(ax); ax.grid(True,axis="x",color="#f0f0f0"); ax.grid(False,axis="y")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#e5e7eb")
    plt.suptitle("Biometric enrolment — state extremes", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "bio_top_bottom.png")

    # 4. National daily trend
    nat = bio.groupby("date")[["bio_total","bio_age_5_17","bio_age_17_plus"]].sum().reset_index()
    nat = nat.sort_values("date")
    nat["smooth"] = nat["bio_total"].rolling(7, min_periods=1, center=True).mean()
    fig, ax = plt.subplots(figsize=(13,5), facecolor=BG)
    ax.set_facecolor(WHITE)
    ax.bar(nat["date"], nat["bio_total"], color=C_BIO, alpha=0.18, width=0.8)
    ax.plot(nat["date"], nat["smooth"], color=C_BIO, linewidth=2.5, label="Bio total (7-day avg)")
    ax.plot(nat["date"],
            nat["bio_age_5_17"].rolling(7,min_periods=1,center=True).mean(),
            color=C_ALT3, linewidth=1.8, linestyle="--", label="Age 5–17")
    ax.plot(nat["date"],
            nat["bio_age_17_plus"].rolling(7,min_periods=1,center=True).mean(),
            color="#9ca3af", linewidth=1.8, linestyle=":", label="Age 17+")
    month_lines(ax, nat["smooth"].max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fmt_y(ax); ax.legend(fontsize=9, loc="upper left")
    ax.set_title("National biometric enrolment — daily total with age breakdown",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    ax_style(ax)
    save(fig, "bio_growth_trend.png")

    # 5. State volatility ranking
    vol = bio.groupby("state")["bio_7day_std"].mean().sort_values(ascending=True).tail(30)
    med = vol.median()
    fig, ax = plt.subplots(figsize=(10, 9), facecolor=BG)
    ax.set_facecolor(WHITE)
    colors = [C_NEG if v > med else "#9ca3af" for v in vol]
    ax.barh(vol.index, vol.values, color=colors, alpha=0.85, height=0.65)
    ax.axvline(med, color=C_BIO, linewidth=1.2, linestyle="--",
               label=f"Median = {med:.2f}")
    ax.legend(fontsize=9)
    ax.set_title("Biometric enrolment volatility by state (7-day std)",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    ax_style(ax)
    save(fig, "bio_volatility_map.png")

    # 6. Age5 ratio vs dependency scatter coloured by growth
    state_agg = bio.groupby("state").agg(
        age5=("age_5_ratio","mean"),
        dep=("dependency_ratio","mean"),
        growth=("bio_growth_pct","mean"),
        total=("bio_total","sum"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9,7), facecolor=BG)
    ax.set_facecolor(WHITE)
    sc = ax.scatter(state_agg["age5"], state_agg["dep"],
                    c=state_agg["growth"], cmap="RdYlGn",
                    s=state_agg["total"]/state_agg["total"].max()*600+40,
                    alpha=0.75, linewidths=0.5, edgecolors="white")
    plt.colorbar(sc, ax=ax, shrink=0.6).set_label("Avg daily growth %", fontsize=9)
    for _, r in state_agg.nlargest(10,"total").iterrows():
        ax.annotate(r["state"][:12], (r["age5"], r["dep"]),
                    xytext=(5,4), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Age 5–17 ratio", fontsize=10)
    ax.set_ylabel("Dependency ratio", fontsize=10)
    ax.set_title("Age 5–17 ratio vs dependency · size=bio total · colour=growth",
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax_style(ax)
    save(fig, "bio_ratio_scatter.png")


# ══════════════════════════════════════════════════════════════════════════
# ENROLMENT CHARTS
# ══════════════════════════════════════════════════════════════════════════

def enrol_charts(enrol):
    print("\n── Enrolment charts ──")

    # 1. National age group pie
    age_totals = {
        "Age 0–5":  enrol["age_0_5"].sum(),
        "Age 5–17": enrol["age_5_17"].sum(),
        "Age 18+":  enrol["age_18_plus"].sum(),
    }
    fig, ax = plt.subplots(figsize=(7,7), facecolor=BG)
    wedges, texts, autotexts = ax.pie(
        age_totals.values(),
        labels=age_totals.keys(),
        autopct="%1.1f%%",
        colors=[C_EN, C_ALT, C_BIO],
        startangle=140,
        wedgeprops={"linewidth":1.5, "edgecolor":"white"},
        pctdistance=0.82,
    )
    for at in autotexts: at.set_fontsize(11); at.set_fontweight("bold")
    for t  in texts:     t.set_fontsize(12)
    total = sum(age_totals.values())
    ax.text(0, 0, f"{total/1e6:.1f}M\ntotal",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#1a1a2e")
    ax.set_title("National enrolment by age group",
                 fontsize=13, fontweight="bold", pad=16)
    save(fig, "enrol_age_pie.png")

    # 2. Adult vs minor stacked bar by state
    state_rat = enrol.groupby("state").agg(
        minor=("minor_ratio","mean"),
        adult=("adult_ratio","mean"),
        total=("enrol_total","sum"),
    ).reset_index().sort_values("adult", ascending=True)
    fig, ax = plt.subplots(figsize=(10,11), facecolor=BG)
    ax.set_facecolor(WHITE)
    ax.barh(state_rat["state"], state_rat["minor"],
            color=C_EN, alpha=0.85, height=0.65, label="Minor (0–17)")
    ax.barh(state_rat["state"], state_rat["adult"],
            color=C_BIO, alpha=0.85, height=0.65,
            left=state_rat["minor"], label="Adult (18+)")
    ax.axvline(0.5, color="#d1d5db", linewidth=0.8, linestyle="--")
    ax.set_xlim(0,1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=10)
    ax.set_title("Adult vs minor enrolment ratio — by state",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.spines["bottom"].set_color("#e5e7eb")
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
    ax.grid(True, axis="x", color="#f0f0f0"); ax.grid(False, axis="y")
    save(fig, "enrol_adult_minor_bar.png")

    # 3. State x month growth heatmap
    pivot = enrol.pivot_table(index="state", columns="month",
                               values="enrol_growth_pct", aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index].head(30)
    fig, ax = plt.subplots(figsize=(14,10), facecolor=BG)
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", center=0,
                linewidths=0.3, linecolor="#f0f0f0",
                annot=True, fmt=".1f", annot_kws={"size":7},
                cbar_kws={"shrink":0.5, "label":"Avg daily growth %"})
    ax.set_title("Enrolment daily growth % — state × month",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8); ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    save(fig, "enrol_growth_heatmap.png")

    # 4. Fastest growing states
    growth = enrol.groupby("state")["enrol_growth_pct"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10,9), facecolor=BG)
    ax.set_facecolor(WHITE)
    med = growth.median()
    colors = [C_EN if v > med else "#9ca3af" for v in growth]
    ax.barh(growth.index, growth.values, color=colors, alpha=0.85, height=0.65)
    ax.axvline(med, color=C_BIO, linewidth=1.2, linestyle="--",
               label=f"Median = {med:.2f}%")
    ax.set_title("Enrolment growth momentum — avg daily % change by state",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.legend(fontsize=9)
    ax_style(ax); ax.grid(True,axis="x",color="#f0f0f0"); ax.grid(False,axis="y")
    save(fig, "enrol_top_growth.png")

    # 5. Volatility ranking
    vol = enrol.groupby("state")["enrol_7day_std"].mean().sort_values(ascending=True).tail(30)
    med = vol.median()
    fig, ax = plt.subplots(figsize=(10,9), facecolor=BG)
    ax.set_facecolor(WHITE)
    ax.barh(vol.index, vol.values,
            color=[C_NEG if v > med else "#9ca3af" for v in vol],
            alpha=0.85, height=0.65)
    ax.axvline(med, color=C_EN, linewidth=1.2, linestyle="--",
               label=f"Median = {med:.1f}")
    ax.set_title("Enrolment volatility by state (7-day std)",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.legend(fontsize=9)
    ax_style(ax); ax.grid(True,axis="x",color="#f0f0f0"); ax.grid(False,axis="y")
    save(fig, "enrol_volatility.png")

    # 6. National daily trend with all 3 age groups
    nat = enrol.groupby("date")[["enrol_total","age_0_5","age_5_17","age_18_plus"]].sum().reset_index().sort_values("date")
    fig, ax = plt.subplots(figsize=(13,5), facecolor=BG)
    ax.set_facecolor(WHITE)
    ax.bar(nat["date"], nat["enrol_total"], color=C_EN, alpha=0.15, width=0.8)
    ax.plot(nat["date"], nat["enrol_total"].rolling(7,min_periods=1,center=True).mean(),
            color=C_EN, linewidth=2.5, label="Total (7-day avg)")
    ax.plot(nat["date"], nat["age_0_5"].rolling(7,min_periods=1,center=True).mean(),
            color=C_ALT, linewidth=1.8, linestyle="--", label="Age 0–5")
    ax.plot(nat["date"], nat["age_5_17"].rolling(7,min_periods=1,center=True).mean(),
            color=C_BIO, linewidth=1.8, linestyle="-.", label="Age 5–17")
    ax.plot(nat["date"], nat["age_18_plus"].rolling(7,min_periods=1,center=True).mean(),
            color="#9ca3af", linewidth=1.8, linestyle=":", label="Age 18+")
    month_lines(ax, nat["enrol_total"].rolling(7,min_periods=1,center=True).mean().max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fmt_y(ax); ax.legend(fontsize=9, loc="upper left")
    ax.set_title("National enrolment — daily total by age group",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    ax_style(ax)
    save(fig, "enrol_trend.png")


# ══════════════════════════════════════════════════════════════════════════
# DEMOGRAPHIC CHARTS
# ══════════════════════════════════════════════════════════════════════════

def demo_charts(demo):
    print("\n── Demographic charts ──")

    # 1. Dependency ratio distribution + stats
    dep = demo.groupby("state")["demo_dependency"].mean()
    fig, axes = plt.subplots(1,2, figsize=(13,5), facecolor=BG)

    ax = axes[0]; ax.set_facecolor(WHITE)
    ax.hist(dep.values, bins=30, color=C_DEM, alpha=0.75,
            edgecolor="white", linewidth=0.4)
    ax.axvline(dep.mean(),   color="#1a1a2e", linewidth=1.5,
               linestyle="--", label=f"Mean = {dep.mean():.3f}")
    ax.axvline(dep.median(), color="#9ca3af", linewidth=1.2,
               linestyle=":",  label=f"Median = {dep.median():.3f}")
    ax.set_xlabel("Dependency ratio", fontsize=10)
    ax.set_title("Demo dependency ratio distribution",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#e5e7eb")

    ax2 = axes[1]; ax2.set_facecolor(WHITE)
    sorted_dep = dep.sort_values()
    colors_d = [C_DEM if v > dep.median() else C_ALT2 for v in sorted_dep]
    ax2.barh(sorted_dep.index, sorted_dep.values,
             color=colors_d, alpha=0.85, height=0.65)
    ax2.axvline(dep.median(), color="#1a1a2e", linewidth=1,
                linestyle="--", alpha=0.6)
    ax2.set_title("Avg dependency ratio — all states",
                  fontsize=12, fontweight="bold", loc="left")
    ax2.tick_params(axis="y", labelsize=7)
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color("#e5e7eb")
    ax2.grid(True, axis="x", color="#f0f0f0"); ax2.grid(False, axis="y")

    plt.tight_layout()
    save(fig, "demo_dependency_dist.png")

    # 2. State comparison — demo total + age group split
    state_agg = demo.groupby("state").agg(
        age5   = ("demo_age_5_17","sum"),
        age17  = ("demo_age_17_plus","sum"),
        total  = ("demo_total","sum"),
        dep    = ("demo_dependency","mean"),
        growth = ("demo_growth_pct","mean"),
    ).reset_index().sort_values("total", ascending=True).tail(30)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG)
    ax.set_facecolor(WHITE)
    ax.barh(state_agg["state"], state_agg["age5"],
            color=C_DEM, alpha=0.85, height=0.65, label="Age 5–17")
    ax.barh(state_agg["state"], state_agg["age17"],
            color=C_ALT2, alpha=0.85, height=0.65,
            left=state_agg["age5"], label="Age 17+")
    fmt_x(ax); ax.legend(fontsize=10)
    ax.grid(True, axis="x", color="#f0f0f0"); ax.grid(False, axis="y")
    ax.set_title("Demographic enrolment by age group — top 30 states",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#e5e7eb")
    save(fig, "demo_state_comparison.png")

    # 3. State x month age5 ratio heatmap
    pivot = demo.pivot_table(index="state", columns="month",
                              values="demo_age5_ratio", aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index].head(30)
    fig, ax = plt.subplots(figsize=(14,10), facecolor=BG)
    sns.heatmap(pivot, ax=ax, cmap="YlOrBr", linewidths=0.3,
                linecolor="#f0f0f0", annot=True, fmt=".2f",
                annot_kws={"size":7},
                cbar_kws={"shrink":0.5, "label":"Age 5–17 ratio"})
    ax.set_title("Demographic age 5–17 ratio — state × month",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8); ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    save(fig, "demo_age_ratio_heatmap.png")

    # 4. National daily trend with age breakdown
    nat = demo.groupby("date")[["demo_total","demo_age_5_17","demo_age_17_plus"]].sum().reset_index().sort_values("date")
    fig, ax = plt.subplots(figsize=(13,5), facecolor=BG)
    ax.set_facecolor(WHITE)
    ax.bar(nat["date"], nat["demo_total"], color=C_DEM, alpha=0.15, width=0.8)
    ax.plot(nat["date"], nat["demo_total"].rolling(7,min_periods=1,center=True).mean(),
            color=C_DEM, linewidth=2.5, label="Total (7-day avg)")
    ax.plot(nat["date"], nat["demo_age_5_17"].rolling(7,min_periods=1,center=True).mean(),
            color=C_ALT2, linewidth=1.8, linestyle="--", label="Age 5–17")
    ax.plot(nat["date"], nat["demo_age_17_plus"].rolling(7,min_periods=1,center=True).mean(),
            color="#9ca3af", linewidth=1.8, linestyle=":", label="Age 17+")
    month_lines(ax, nat["demo_total"].rolling(7,min_periods=1,center=True).mean().max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fmt_y(ax); ax.legend(fontsize=9, loc="upper left")
    ax.set_title("National demographic enrolment — daily total by age group",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    ax_style(ax)
    save(fig, "demo_trend.png")


# ══════════════════════════════════════════════════════════════════════════
# COMBINED CHARTS
# ══════════════════════════════════════════════════════════════════════════

def combined_charts(bio, enrol, demo):
    print("\n── Combined charts ──")

    # 1. All 3 tables national trend on same chart
    bio_nat   = bio.groupby("date")["bio_total"].sum().reset_index()
    enrol_nat = enrol.groupby("date")["enrol_total"].sum().reset_index()
    demo_nat  = demo.groupby("date")["demo_total"].sum().reset_index()

    merged = bio_nat.merge(enrol_nat, on="date").merge(demo_nat, on="date")
    merged = merged.sort_values("date")

    fig, ax = plt.subplots(figsize=(13,6), facecolor=BG)
    ax.set_facecolor(WHITE)
    for col, color, label in [
        ("bio_total",   C_BIO, "Biometric"),
        ("enrol_total", C_EN,  "Enrolment"),
        ("demo_total",  C_DEM, "Demographic"),
    ]:
        smooth = merged[col].rolling(7, min_periods=1, center=True).mean()
        ax.plot(merged["date"], smooth, color=color, linewidth=2.3, label=label)
        ax.fill_between(merged["date"], smooth, alpha=0.07, color=color)

    month_lines(ax, merged[["bio_total","enrol_total","demo_total"]].max().max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fmt_y(ax); ax.legend(fontsize=10, loc="upper left")
    ax.set_title("All 3 tables — national daily total (7-day smoothed)",
                 fontsize=13, fontweight="bold", loc="left", pad=10)
    ax_style(ax)
    save(fig, "all_tables_trend.png")

    # 2. Feature correlation heatmap
    state_merged = (
        bio.groupby("state").agg(
            bio_total    =("bio_total","mean"),
            bio_dep      =("dependency_ratio","mean"),
            bio_age5     =("age_5_ratio","mean"),
            bio_growth   =("bio_growth_pct","mean"),
            bio_vol      =("bio_7day_std","mean"),
        ).reset_index()
        .merge(
            enrol.groupby("state").agg(
                enrol_total  =("enrol_total","mean"),
                minor_ratio  =("minor_ratio","mean"),
                adult_ratio  =("adult_ratio","mean"),
                enrol_growth =("enrol_growth_pct","mean"),
                enrol_vol    =("enrol_7day_std","mean"),
            ).reset_index(), on="state")
        .merge(
            demo.groupby("state").agg(
                demo_total   =("demo_total","mean"),
                demo_dep     =("demo_dependency","mean"),
                demo_age5    =("demo_age5_ratio","mean"),
                demo_growth  =("demo_growth_pct","mean"),
            ).reset_index(), on="state")
    )

    num_cols = state_merged.select_dtypes("float").columns
    corr = state_merged[num_cols].corr()

    fig, ax = plt.subplots(figsize=(13,11), facecolor=BG)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, cmap="RdYlGn", center=0,
                vmin=-1, vmax=1, linewidths=0.4, linecolor="#f0f0f0",
                annot=True, fmt=".2f", annot_kws={"size":7},
                cbar_kws={"shrink":0.6, "label":"Pearson r"})
    ax.set_title("Feature correlation — Bio × Enrolment × Demographic (state averages)",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    save(fig, "correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main(db_path):
    print("="*60)
    print("TABLE ANALYSIS — COMPLETE VISUALIZATION")
    print("="*60)
    setup()
    print("\n── Fetching data ──")
    bio, enrol, demo = fetch(db_path)
    bio_charts(bio)
    enrol_charts(enrol)
    demo_charts(demo)
    combined_charts(bio, enrol, demo)
    print("\n" + "="*60)
    print("DONE — table_output/")
    print("  BIOMETRIC:    bio_age_distribution, bio_state_heatmap,")
    print("                bio_top_bottom, bio_growth_trend,")
    print("                bio_volatility_map, bio_ratio_scatter")
    print("  ENROLMENT:    enrol_age_pie, enrol_adult_minor_bar,")
    print("                enrol_growth_heatmap, enrol_top_growth,")
    print("                enrol_volatility, enrol_trend")
    print("  DEMOGRAPHIC:  demo_dependency_dist, demo_state_comparison,")
    print("                demo_age_ratio_heatmap, demo_trend")
    print("  COMBINED:     all_tables_trend, correlation_heatmap")
    print("="*60)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    main(p.parse_args().db)