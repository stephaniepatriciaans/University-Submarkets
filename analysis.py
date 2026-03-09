from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

COLOR_MAP = {
    "SDSU": "#C41230",
    "UCSD": "#00629B",
    "USD": "#2E8B57",
}

SCHOOL_TO_SHORT = {
    "San Diego State University": "SDSU",
    "UC San Diego": "UCSD",
    "University of San Diego": "USD",
    "SDSU": "SDSU",
    "UCSD": "UCSD",
    "USD": "USD",
}

POSSIBLE_FHFA_FILES = [
    DATA_DIR / "fhfa_2022_2024.csv",
    DATA_DIR / "fhfa_2022-2024.csv",
    DATA_DIR / "fhfa_full_history.csv",
]


def get_short_name(label: str) -> str:
    return SCHOOL_TO_SHORT.get(str(label), str(label))


def get_color(label: str) -> str:
    return COLOR_MAP.get(get_short_name(label), "#444444")


def get_bar_colors(labels) -> list[str]:
    return [get_color(label) for label in labels]


def build_starter_dataset() -> pd.DataFrame:
    data = [
        {
            "school": "San Diego State University",
            "short_name": "SDSU",
            "proxy_zip": "92115",
            "area_name": "College Area",
            "total_enrollment": 39000,
            "undergrad_enrollment": np.nan,
            "grad_enrollment": np.nan,
            "avg_home_value_usd": 835518,
            "avg_rent_usd": 4300,
            "direct_trolley_access": 1,
            "transit_score": 1.0,
            "current_housing_beds": np.nan,
            "future_housing_beds": 5220,
            "net_new_beds": 4468,
            "expansion_score": 0.85,
            "notes": "Direct Green Line access; strong housing expansion signal.",
        },
        {
            "school": "UC San Diego",
            "short_name": "UCSD",
            "proxy_zip": "92122",
            "area_name": "University City / UTC",
            "total_enrollment": 44256,
            "undergrad_enrollment": 34955,
            "grad_enrollment": np.nan,
            "avg_home_value_usd": 1072929,
            "avg_rent_usd": 3009,
            "direct_trolley_access": 1,
            "transit_score": 1.0,
            "current_housing_beds": 19710,
            "future_housing_beds": 38620,
            "net_new_beds": 18910,
            "expansion_score": 1.0,
            "notes": "Blue Line access; strongest long-run official expansion story.",
        },
        {
            "school": "University of San Diego",
            "short_name": "USD",
            "proxy_zip": "92110",
            "area_name": "Linda Vista / Morena",
            "total_enrollment": 9714,
            "undergrad_enrollment": 5851,
            "grad_enrollment": 3863,
            "avg_home_value_usd": 989256,
            "avg_rent_usd": 2833,
            "direct_trolley_access": 0,
            "transit_score": 0.35,
            "current_housing_beds": np.nan,
            "future_housing_beds": np.nan,
            "net_new_beds": np.nan,
            "expansion_score": 0.20,
            "notes": "Useful benchmark, but weaker direct rail and expansion signal.",
        },
    ]

    df = pd.DataFrame(data)
    df["annual_rent_usd"] = df["avg_rent_usd"] * 12
    df["rent_to_value_ratio_pct"] = (df["annual_rent_usd"] / df["avg_home_value_usd"]) * 100
    return df


def minmax_scale(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    min_val = s.min()
    max_val = s.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(np.full(len(s), 0.5), index=s.index)

    scaled = (s - min_val) / (max_val - min_val)
    if not higher_is_better:
        scaled = 1 - scaled
    return scaled.fillna(0.5)


def score_submarkets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["score_enrollment"] = minmax_scale(out["total_enrollment"], higher_is_better=True)
    out["score_affordability"] = minmax_scale(out["avg_home_value_usd"], higher_is_better=False)
    out["score_income_support"] = minmax_scale(out["rent_to_value_ratio_pct"], higher_is_better=True)
    out["score_transit"] = pd.to_numeric(out["transit_score"], errors="coerce").fillna(0.5)
    out["score_expansion"] = pd.to_numeric(out["expansion_score"], errors="coerce").fillna(0.5)

    out["investment_score"] = (
        0.25 * out["score_enrollment"]
        + 0.20 * out["score_affordability"]
        + 0.20 * out["score_income_support"]
        + 0.15 * out["score_transit"]
        + 0.20 * out["score_expansion"]
    )

    out = out.sort_values(["investment_score", "total_enrollment"], ascending=[False, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def find_fhfa_file() -> Path | None:
    for path in POSSIBLE_FHFA_FILES:
        if path.exists():
            return path
    return None


def load_fhfa_history(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"proxy_zip", "school", "year", "fhfa_hpi_native_base"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {sorted(missing)}")

    df = df.copy()
    df["proxy_zip"] = df["proxy_zip"].astype(str).str.zfill(5)
    df["zip_code"] = df["proxy_zip"]
    df["school"] = df["school"].astype(str)
    df["short_name"] = df["school"].map(get_short_name).fillna(df["school"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["hpi"] = pd.to_numeric(df["fhfa_hpi_native_base"], errors="coerce")

    if "index_2022_eq_100" in df.columns:
        df["index_2022_eq_100"] = pd.to_numeric(df["index_2022_eq_100"], errors="coerce")

    if "annual_change_pct" in df.columns:
        df["annual_change_pct"] = pd.to_numeric(df["annual_change_pct"], errors="coerce")

    df = df.dropna(subset=["year", "hpi"]).copy()
    df["year"] = df["year"].astype(int)

    return df.sort_values(["short_name", "year"]).reset_index(drop=True)


def save_summary_table(scored: pd.DataFrame) -> None:
    summary = scored[
        [
            "rank",
            "short_name",
            "school",
            "proxy_zip",
            "area_name",
            "total_enrollment",
            "avg_home_value_usd",
            "avg_rent_usd",
            "rent_to_value_ratio_pct",
            "direct_trolley_access",
            "net_new_beds",
            "investment_score",
            "notes",
        ]
    ].copy()

    summary["investment_score"] = summary["investment_score"].round(3)
    summary["rent_to_value_ratio_pct"] = summary["rent_to_value_ratio_pct"].round(2)
    summary.to_csv(TABLES_DIR / "summary_table.csv", index=False)


def save_analysis_summary_table(scored: pd.DataFrame) -> None:
    table = scored[
        [
            "rank",
            "short_name",
            "total_enrollment",
            "net_new_beds",
            "direct_trolley_access",
            "avg_home_value_usd",
            "rent_to_value_ratio_pct",
            "investment_score",
        ]
    ].copy()

    table = table.rename(
        columns={
            "short_name": "school",
            "total_enrollment": "enrollment",
            "direct_trolley_access": "direct_trolley",
        }
    )

    table["rent_to_value_ratio_pct"] = table["rent_to_value_ratio_pct"].round(2)
    table["investment_score"] = table["investment_score"].round(3)
    table["avg_home_value_usd"] = table["avg_home_value_usd"].round(0)
    table.to_csv(TABLES_DIR / "analysis_summary_table.csv", index=False)


def save_fhfa_tables(fhfa_df: pd.DataFrame) -> None:
    fhfa_df.to_csv(TABLES_DIR / "fhfa_history_cleaned.csv", index=False)

    latest = (
        fhfa_df.sort_values(["short_name", "year"])
        .groupby("short_name", as_index=False)
        .tail(1)
        .copy()
    )

    latest["hpi"] = latest["hpi"].round(2)

    if "annual_change_pct" in latest.columns:
        latest["annual_change_pct"] = latest["annual_change_pct"].round(2)

    if "index_2022_eq_100" in latest.columns:
        latest["index_2022_eq_100"] = latest["index_2022_eq_100"].round(2)

    latest.to_csv(TABLES_DIR / "fhfa_latest_summary.csv", index=False)


def save_bar_chart(
    labels,
    values,
    title: str,
    xlabel: str,
    ylabel: str,
    output_name: str,
) -> None:
    colors = get_bar_colors(labels)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=colors, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=200)
    plt.close()


def make_home_value_chart(scored: pd.DataFrame) -> None:
    save_bar_chart(
        labels=scored["short_name"],
        values=scored["avg_home_value_usd"],
        title="University Submarket Home Value Comparison",
        xlabel="University Submarket",
        ylabel="Average Home Value ($)",
        output_name="home_value_comparison.png",
    )


def make_rent_to_value_chart(scored: pd.DataFrame) -> None:
    save_bar_chart(
        labels=scored["short_name"],
        values=scored["rent_to_value_ratio_pct"],
        title="Annual Rent-to-Home-Value Ratio",
        xlabel="University Submarket",
        ylabel="Annual Rent / Home Value (%)",
        output_name="rent_to_value_ratio.png",
    )


def make_investment_score_chart(scored: pd.DataFrame) -> None:
    save_bar_chart(
        labels=scored["short_name"],
        values=scored["investment_score"],
        title="University Submarket Investment Screening Score",
        xlabel="University Submarket",
        ylabel="Screening Score (0-1)",
        output_name="investment_score_ranking.png",
    )


def make_enrollment_chart(scored: pd.DataFrame) -> None:
    save_bar_chart(
        labels=scored["short_name"],
        values=scored["total_enrollment"],
        title="University Enrollment Comparison",
        xlabel="University",
        ylabel="Total Enrollment",
        output_name="enrollment_comparison.png",
    )


def make_expansion_chart(scored: pd.DataFrame) -> None:
    expansion_df = scored[["short_name", "net_new_beds"]].copy()
    expansion_df["net_new_beds"] = expansion_df["net_new_beds"].fillna(0)

    save_bar_chart(
        labels=expansion_df["short_name"],
        values=expansion_df["net_new_beds"],
        title="Planned Net New Student Housing Beds",
        xlabel="University",
        ylabel="Net New Beds",
        output_name="expansion_housing_comparison.png",
    )


def make_value_vs_yield_scatter(scored: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))

    for _, row in scored.iterrows():
        x = row["avg_home_value_usd"]
        y = row["rent_to_value_ratio_pct"]
        label = row["short_name"]

        plt.scatter(
            x,
            y,
            s=180,
            color=get_color(label),
            alpha=0.9,
            edgecolor="black",
        )
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            ha="left",
            fontsize=11,
        )

    plt.title("Investment Tradeoff: Home Value vs Rent-to-Value Ratio")
    plt.xlabel("Average Home Value ($)")
    plt.ylabel("Annual Rent / Home Value (%)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "value_vs_yield_scatter.png", dpi=200)
    plt.close()


def make_fhfa_hpi_line_chart(fhfa_df: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 6))

    for short_name, group in fhfa_df.groupby("short_name"):
        group = group.sort_values("year")
        plt.plot(
            group["year"],
            group["hpi"],
            marker="o",
            linewidth=2.5,
            color=get_color(short_name),
            label=short_name,
        )

    plt.title("FHFA ZIP5 House Price Index")
    plt.xlabel("Year")
    plt.ylabel("FHFA HPI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fhfa_hpi_line.png", dpi=200)
    plt.close()


def make_fhfa_index_line_chart(fhfa_df: pd.DataFrame) -> None:
    if "index_2022_eq_100" not in fhfa_df.columns:
        return

    usable = fhfa_df.dropna(subset=["index_2022_eq_100"]).copy()
    if usable.empty:
        return

    plt.figure(figsize=(11, 6))

    for short_name, group in usable.groupby("short_name"):
        group = group.sort_values("year")
        plt.plot(
            group["year"],
            group["index_2022_eq_100"],
            marker="o",
            linewidth=2.5,
            color=get_color(short_name),
            label=short_name,
        )

    plt.title("Indexed FHFA Price Growth (2022 = 100)")
    plt.xlabel("Year")
    plt.ylabel("Index (2022 = 100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fhfa_index_line.png", dpi=200)
    plt.close()


def make_fhfa_annual_change_chart(fhfa_df: pd.DataFrame) -> None:
    if "annual_change_pct" not in fhfa_df.columns:
        return

    latest = (
        fhfa_df.dropna(subset=["annual_change_pct"])
        .sort_values(["short_name", "year"])
        .groupby("short_name", as_index=False)
        .tail(1)
        .copy()
    )

    if latest.empty:
        return

    save_bar_chart(
        labels=latest["short_name"],
        values=latest["annual_change_pct"],
        title="Latest Annual FHFA HPI Change",
        xlabel="University Submarket",
        ylabel="Annual Change (%)",
        output_name="fhfa_latest_annual_change.png",
    )


def print_assignment_summary(scored: pd.DataFrame, fhfa_df: pd.DataFrame | None = None) -> None:
    top = scored.iloc[0]
    second = scored.iloc[1]

    summary = f"""
    FINAL PROJECT ANSWER

    Proposed geographic market:
    Residential submarkets bordering major San Diego universities, especially SDSU and UCSD.

    Why this market:
    These submarkets combine durable student demand, university expansion, and transit access.
    The evidence suggests these neighborhoods are more attractive than a weaker benchmark like USD.

    Main result:
    The top-ranked starter market is {top['school']} ({top['short_name']}), followed by {second['school']} ({second['short_name']}).

    Interpretation:
    {top['short_name']} scores well because it combines a relatively lower entry price with a strong rent-to-value ratio
    and direct transit access. UCSD remains strategically attractive because of its very large enrollment base and strong
    official long-run expansion story.

    Recommendation:
    A REIT should prioritize university-bordering residential submarkets near SDSU and UCSD, with USD used mainly as a benchmark.
    """

    print(textwrap.dedent(summary).strip())

    if fhfa_df is not None:
        print("\nReal-data extension included:")
        print("- Official FHFA ZIP5 HPI history")
        print("- Indexed growth comparison with 2022 = 100")
        print("- Latest annual HPI change comparison")


def print_saved_files(fhfa_df: pd.DataFrame | None, fhfa_path: Path | None) -> None:
    print("\nSaved files:")
    print(f"- {TABLES_DIR / 'university_submarkets_starter.csv'}")
    print(f"- {TABLES_DIR / 'university_submarkets_scored.csv'}")
    print(f"- {FIGURES_DIR / 'home_value_comparison.png'}")
    print(f"- {FIGURES_DIR / 'rent_to_value_ratio.png'}")
    print(f"- {FIGURES_DIR / 'investment_score_ranking.png'}")
    print(f"- {FIGURES_DIR / 'enrollment_comparison.png'}")
    print(f"- {FIGURES_DIR / 'expansion_housing_comparison.png'}")
    print(f"- {FIGURES_DIR / 'value_vs_yield_scatter.png'}")
    print(f"- {TABLES_DIR / 'summary_table.csv'}")
    print(f"- {TABLES_DIR / 'analysis_summary_table.csv'}")

    if fhfa_df is not None and fhfa_path is not None:
        print(f"- {TABLES_DIR / 'fhfa_history_cleaned.csv'}")
        print(f"- {TABLES_DIR / 'fhfa_latest_summary.csv'}")
        print(f"- {FIGURES_DIR / 'fhfa_hpi_line.png'}")
        print(f"- {FIGURES_DIR / 'fhfa_index_line.png'}")
        print(f"- {FIGURES_DIR / 'fhfa_latest_annual_change.png'}")
        print(f"\nFHFA source file used: {fhfa_path.name}")
    else:
        print("\nNo FHFA CSV found.")
        print("Tried these file names:")
        for path in POSSIBLE_FHFA_FILES:
            print(f"- {path}")


def main() -> None:
    starter = build_starter_dataset()
    scored = score_submarkets(starter)

    starter_path = TABLES_DIR / "university_submarkets_starter.csv"
    scored_path = TABLES_DIR / "university_submarkets_scored.csv"

    starter.to_csv(starter_path, index=False)
    scored.to_csv(scored_path, index=False)

    make_home_value_chart(scored)
    make_rent_to_value_chart(scored)
    make_investment_score_chart(scored)
    make_enrollment_chart(scored)
    make_expansion_chart(scored)
    make_value_vs_yield_scatter(scored)

    save_summary_table(scored)
    save_analysis_summary_table(scored)

    fhfa_path = find_fhfa_file()
    fhfa_df = None

    if fhfa_path is not None:
        print(f"Using FHFA file: {fhfa_path.name}")
        fhfa_df = load_fhfa_history(fhfa_path)
        save_fhfa_tables(fhfa_df)
        make_fhfa_hpi_line_chart(fhfa_df)
        make_fhfa_index_line_chart(fhfa_df)
        make_fhfa_annual_change_chart(fhfa_df)

    print_assignment_summary(scored, fhfa_df)
    print_saved_files(fhfa_df, fhfa_path)


if __name__ == "__main__":
    main()
