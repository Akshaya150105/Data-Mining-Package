"""
time_series.py
==============
Generates time series trend charts from preprocessed DuckDB tables.
Uses all 70 common dates across bio/demo/enrolment tables.

Outputs (in timeseries_output/):
    ts_state_total.png          total enrolment per state over time
    ts_top10_states.png         top 10 states smoothed trend lines
    ts_heatmap.png              state x date enrolment heatmap
    ts_monthly_growth.png       month-over-month growth rate per state
    ts_volatility.png           enrolment volatility ranking
    ts_data.csv                 full state x date pivot table

Usage:
    python time_series.py --db ../database/aadhar.duckdb
"""

import argparse, warnings
from pathlib import Path
import duckdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")
OUTPUT_DIR = Path("timeseries_output")
OUTPUT_DIR.mkdir(exist_ok=True)

BG    = "#f9f8f6"
WHITE = "#ffffff"
COLORS = [
    "#1D9E75","#534AB7","#D85A30","#BA7517","#D4537E",
    "#378ADD","#639922","#E24B4A","#5DCAA5","#7F77DD",
    "#F0997B","#9FE1CB","#FAC775","#F4C0D1","#B5D4F4",
]

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   WHITE,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.color":       "#f0f0f0",
        "grid.linewidth":   0.8,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
    })

# ══════════════════════════════════════════════════════════════════════════
# FETCH
# ══════════════════════════════════════════════════════════════════════════

def fetch_timeseries(db_path: str):
    """
    Returns a daily state-level time series dataframe.
    Aggregates enrolment_data_preprocessed → SUM(enrol_total) per state per date.
    Also fetches bio and demo totals for comparison.
    """
    con = duckdb.connect(db_path, read_only=True)

    # Auto-detect all common dates
    bio_d   = set(con.execute("SELECT DISTINCT CAST(date AS DATE)::VARCHAR FROM biometric_data_preprocessed").df().iloc[:,0])
    demo_d  = set(con.execute("SELECT DISTINCT CAST(date AS DATE)::VARCHAR FROM demographic_data_preprocessed").df().iloc[:,0])
    enrol_d = set(con.execute("SELECT DISTINCT CAST(date AS DATE)::VARCHAR FROM enrolment_data_preprocessed").df().iloc[:,0])
    all_dates = sorted(bio_d & demo_d & enrol_d)
    print(f"  Common dates found: {len(all_dates)}  ({all_dates[0]} → {all_dates[-1]})")

    dts = ", ".join([f"'{d}'" for d in all_dates])

    # Enrolment totals per state per date
    enrol = con.execute(f"""
        SELECT
            CAST(date AS DATE)::VARCHAR AS date,
            state,
            SUM(enrol_total)           AS enrol_total,
            AVG(daily_pct_change)      AS avg_growth,
            AVG(enrol_total_7day_avg)  AS rolling_avg,
            AVG(enrol_total_7day_std)  AS volatility
        FROM enrolment_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts})
          AND state IS NOT NULL
        GROUP BY date, state
        ORDER BY date, state
    """).fetchdf()

    # Bio totals per state per date (for overlay)
    bio = con.execute(f"""
        SELECT
            CAST(date AS DATE)::VARCHAR AS date,
            state,
            SUM(bio_total) AS bio_total
        FROM biometric_data_preprocessed
        WHERE CAST(date AS DATE)::VARCHAR IN ({dts})
          AND state IS NOT NULL
        GROUP BY date, state
        ORDER BY date, state
    """).fetchdf()

    con.close()

    enrol["date"] = pd.to_datetime(enrol["date"])
    bio["date"]   = pd.to_datetime(bio["date"])
    enrol["state"] = enrol["state"].str.strip()
    bio["state"]   = bio["state"].str.strip()

    df = enrol.merge(bio, on=["date","state"], how="left")
    df = df[df["state"] != ""]
    print(f"  States: {df['state'].nunique()}  |  Rows: {len(df):,}")
    return df, all_dates


# ══════════════════════════════════════════════════════════════════════════
# CHART 1 — Top 10 states: smoothed enrolment trend lines
# ══════════════════════════════════════════════════════════════════════════

def plot_top10_trends(df: pd.DataFrame):
    # Rank states by total enrolment
    top10 = (df.groupby("state")["enrol_total"].sum()
               .nlargest(10).index.tolist())

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    ax.set_facecolor(WHITE)

    for i, state in enumerate(top10):
        sdf = df[df["state"] == state].sort_values("date")
        # 3-point rolling smooth for visual clarity
        smoothed = sdf["enrol_total"].rolling(3, min_periods=1, center=True).mean()
        ax.plot(sdf["date"], smoothed,
                color=COLORS[i % len(COLORS)],
                linewidth=2, label=state[:20], alpha=0.85)
        # Mark the actual data points (sparse for monthly)
        ax.scatter(sdf["date"], sdf["enrol_total"],
                   color=COLORS[i % len(COLORS)], s=18, alpha=0.5, zorder=3)

    # Month separator lines
    for month_start in ["2025-09-01","2025-10-01","2025-11-01","2025-12-01"]:
        ax.axvline(pd.Timestamp(month_start), color="#e5e7eb",
                   linewidth=1, linestyle="--", alpha=0.8)
        ax.text(pd.Timestamp(month_start), ax.get_ylim()[1] * 0.02,
                f" {pd.Timestamp(month_start).strftime('%b')}",
                fontsize=8, color="#9ca3af")

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Total enrolment", fontsize=10)
    ax.set_title("Top 10 states — Aadhaar enrolment trend (Apr–Dec 2025)",
                 fontsize=14, fontweight="bold", pad=14, loc="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}K" if x < 1e6 else f"{x/1e6:.1f}M"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.xticks(rotation=35, ha="right")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9,
              bbox_to_anchor=(0, 1))
    ax.spines["bottom"].set_color("#e5e7eb")
    ax.spines["left"].set_color("#e5e7eb")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ts_top10_states.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → ts_top10_states.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 2 — State × date heatmap (enrolment intensity)
# ══════════════════════════════════════════════════════════════════════════

def plot_heatmap(df: pd.DataFrame):
    # Pivot: rows = state, cols = date
    pivot = df.pivot_table(index="state", columns="date",
                           values="enrol_total", aggfunc="sum")

    # Keep top 30 states by mean enrolment
    pivot = pivot.loc[pivot.mean(axis=1).nlargest(30).index]

    # Normalise each state to [0,1] so scale differences don't drown small states
    pivot_norm = pivot.div(pivot.max(axis=1), axis=0)

    # Shorten column labels to "Sep 1", "Oct 13" etc.
    col_labels = [d.strftime("%d %b") if hasattr(d, 'strftime')
                  else str(d) for d in pivot_norm.columns]

    fig, ax = plt.subplots(figsize=(18, 9), facecolor=BG)
    sns.heatmap(
        pivot_norm, ax=ax,
        cmap="YlOrRd",
        xticklabels=col_labels,
        yticklabels=pivot_norm.index,
        linewidths=0,
        cbar_kws={"shrink": 0.4, "label": "Normalised enrolment (0=min, 1=max per state)"},
    )
    ax.set_title("Enrolment intensity — state × date (row-normalised)",
                 fontsize=14, fontweight="bold", pad=14, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=9)

    # Add month separator lines
    cols = list(pivot_norm.columns)
    for sep_date in [pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01"),
                     pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-01")]:
        try:
            idx = next(i for i, c in enumerate(cols) if c >= sep_date)
            ax.axvline(idx, color="white", linewidth=2)
            ax.text(idx + 0.3, -1.2, sep_date.strftime("%b"),
                    fontsize=8, color="#6b7280")
        except StopIteration:
            pass

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ts_heatmap.png",
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → ts_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 3 — Month-over-month growth rate (top 20 states)
# ══════════════════════════════════════════════════════════════════════════

def plot_monthly_growth(df: pd.DataFrame):
    # Aggregate to monthly totals per state
    df2 = df.copy()
    df2["month"] = df2["date"].dt.to_period("M")
    monthly = (df2.groupby(["state","month"])["enrol_total"]
                  .sum().reset_index())
    monthly["month_ts"] = monthly["month"].dt.to_timestamp()
    monthly = monthly.sort_values(["state","month_ts"])

    # Calculate MoM % change
    monthly["mom_pct"] = (monthly.groupby("state")["enrol_total"]
                                  .pct_change() * 100)
    monthly = monthly.dropna(subset=["mom_pct"])

    # Top 20 states by total enrolment
    top20 = (df.groupby("state")["enrol_total"].sum()
               .nlargest(20).index.tolist())
    monthly = monthly[monthly["state"].isin(top20)]

    # Pivot for grouped bar chart
    pivot = monthly.pivot_table(index="state", columns="month_ts",
                                values="mom_pct", aggfunc="mean")
    pivot = pivot.sort_values(pivot.columns[-1], ascending=False)

    months = pivot.columns
    x      = np.arange(len(pivot))
    width  = 0.8 / len(months)

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    ax.set_facecolor(WHITE)

    month_colors = ["#1D9E75","#534AB7","#D85A30","#BA7517","#D4537E"]
    for i, (month, col) in enumerate(zip(months, month_colors)):
        vals = pivot[month].fillna(0)
        bars = ax.bar(x + i * width, vals, width,
                      label=month.strftime("%b %Y"),
                      color=col, alpha=0.82)

    ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="-")
    ax.set_xticks(x + width * (len(months) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Month-over-month growth %", fontsize=10)
    ax.set_title("Month-over-month enrolment growth — top 20 states",
                 fontsize=14, fontweight="bold", pad=14, loc="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines["bottom"].set_color("#e5e7eb")
    ax.spines["left"].set_color("#e5e7eb")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ts_monthly_growth.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → ts_monthly_growth.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 4 — Volatility ranking (7-day std per state)
# ══════════════════════════════════════════════════════════════════════════

def plot_volatility(df: pd.DataFrame):
    vol = (df.groupby("state")["volatility"]
             .mean()
             .sort_values(ascending=True)
             .tail(30))

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG)
    ax.set_facecolor(WHITE)

    colors = ["#D85A30" if v > vol.median() else "#9ca3af" for v in vol]
    ax.barh(vol.index, vol.values, color=colors, alpha=0.85, height=0.65)

    med = vol.median()
    ax.axvline(med, color="#534AB7", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.text(med, len(vol) - 0.5, f" median\n {med:.1f}",
            fontsize=8, color="#534AB7", va="top")

    ax.set_xlabel("Avg 7-day enrolment volatility (std)", fontsize=10)
    ax.set_title("Enrolment volatility by state\n(higher = more irregular enrolment pattern)",
                 fontsize=13, fontweight="bold", pad=14, loc="left")
    ax.spines["bottom"].set_color("#e5e7eb")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ts_volatility.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → ts_volatility.png")


# ══════════════════════════════════════════════════════════════════════════
# CHART 5 — National daily total (aggregate across all states)
# ══════════════════════════════════════════════════════════════════════════

def plot_national_trend(df: pd.DataFrame):
    national = (df.groupby("date")[["enrol_total","bio_total"]]
                  .sum().reset_index().sort_values("date"))

    # 7-day rolling average
    national["enrol_smooth"] = national["enrol_total"].rolling(7, min_periods=1, center=True).mean()
    national["bio_smooth"]   = national["bio_total"].rolling(7, min_periods=1, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    ax.set_facecolor(WHITE)

    # Raw bars (light)
    ax.bar(national["date"], national["enrol_total"],
           color="#1D9E75", alpha=0.25, width=0.8, label="_nolegend_")

    # Smoothed lines
    ax.plot(national["date"], national["enrol_smooth"],
            color="#1D9E75", linewidth=2.5, label="Enrolment (7-day avg)")
    ax.plot(national["date"], national["bio_smooth"],
            color="#534AB7", linewidth=2, linestyle="--",
            label="Biometric (7-day avg)", alpha=0.8)

    # Month separators
    for ms in ["2025-09-01","2025-10-01","2025-11-01","2025-12-01"]:
        ax.axvline(pd.Timestamp(ms), color="#e5e7eb", linewidth=1.2, zorder=1)
        ax.text(pd.Timestamp(ms), national["enrol_smooth"].max() * 1.01,
                f" {pd.Timestamp(ms).strftime('%b')}",
                fontsize=9, color="#9ca3af")

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("National daily total", fontsize=10)
    ax.set_title("National Aadhaar enrolment — daily total with 7-day smoothing",
                 fontsize=14, fontweight="bold", pad=14, loc="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.xticks(rotation=35, ha="right")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.spines["bottom"].set_color("#e5e7eb")
    ax.spines["left"].set_color("#e5e7eb")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ts_national.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → ts_national.png")


# ══════════════════════════════════════════════════════════════════════════
# SAVE DATA CSV
# ══════════════════════════════════════════════════════════════════════════

def save_data(df: pd.DataFrame):
    pivot = (df.pivot_table(index="state", columns="date",
                            values="enrol_total", aggfunc="sum")
               .round(0))
    pivot.columns = [str(c.date()) for c in pivot.columns]
    pivot.to_csv(OUTPUT_DIR / "ts_data.csv")
    print(f"  Saved → ts_data.csv  ({len(pivot)} states × {len(pivot.columns)} dates)")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main(db_path: str):
    print("="*60)
    print("TIME SERIES ANALYSIS")
    print("="*60)
    setup_style()

    print("\n── Fetching data ──")
    df, all_dates = fetch_timeseries(db_path)

    print("\n── Generating charts ──")
    plot_national_trend(df)
    plot_top10_trends(df)
    plot_heatmap(df)
    plot_monthly_growth(df)
    plot_volatility(df)
    save_data(df)

    print("\n" + "="*60)
    print("DONE — timeseries_output/")
    print("  ts_national.png       national daily total + 7-day smoothing")
    print("  ts_top10_states.png   top 10 states trend lines")
    print("  ts_heatmap.png        state × date intensity heatmap")
    print("  ts_monthly_growth.png month-over-month growth per state")
    print("  ts_volatility.png     enrolment volatility ranking")
    print("  ts_data.csv           full state × date pivot table")
    print("="*60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    a = p.parse_args()
    main(a.db)