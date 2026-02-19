import streamlit as st
import pandas as pd
import os
import glob
import importlib
import traceback
from datetime import datetime

# -------------------------------------------------
# CONFIG / PATHS
# -------------------------------------------------
st.set_page_config(page_title="Corners Pricing Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOUT_DIR = os.path.join(BASE_DIR, "dout")
os.makedirs(DOUT_DIR, exist_ok=True)

LEAGUES = ["EPL", "LL", "SA", "BL", "L1"]

# Promeni ako ti se model fajl zove drugačije (bez .py)
MODEL_MODULE_NAME = "bookmaker_kornerv12_multileague"


# -------------------------------------------------
# Helpers: file discovery & loading
# -------------------------------------------------
def list_files(pattern: str):
    return sorted(glob.glob(os.path.join(DOUT_DIR, pattern)), key=os.path.getmtime, reverse=True)

def latest_file(pattern: str):
    files = list_files(pattern)
    return files[0] if files else None

def load_csv(path: str):
    return pd.read_csv(path)

def safe_metric(df_sanity: pd.DataFrame, metric_name: str):
    try:
        v = df_sanity.loc[df_sanity["metric"] == metric_name, "value"].values
        return float(v[0]) if len(v) else None
    except Exception:
        return None


# -------------------------------------------------
# Helpers: run model
# -------------------------------------------------
def run_model(
    league: str,
    margin: float,
    over_price_boost: float,
    alpha_squeeze: float,
    n_sims: int,
):
    """
    Imports the model module and runs main().
    Assumes model writes outputs into DOUT_DIR itself.
    """
    mod = importlib.import_module(MODEL_MODULE_NAME)

    # In case you want to override sims quickly without editing model defaults:
    # call main(...) with n_sims
    mod.main(
        league=league,
        margin=margin,
        over_price_boost=over_price_boost,
        alpha_squeeze=alpha_squeeze,
        n_sims=n_sims,
    )


# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.title("⚽ Corners Pricing Dashboard (Multi-league)")

with st.sidebar:
    st.header("⚙️ Controls")

    league = st.selectbox("League", LEAGUES, index=0)

    st.subheader("Run parameters")
    margin = st.slider("Margin", 0.00, 0.12, 0.08, 0.005)
    over_boost = st.slider("Over price boost", 0.00, 0.10, 0.01, 0.005)
    alpha_squeeze = st.slider("Alpha squeeze", 0.50, 1.00, 0.68, 0.01)

    sims_mode = st.selectbox("Sims mode", ["FAST (50k)", "STD (200k)", "HQ (600k)"], index=1)
    n_sims = 50_000 if sims_mode.startswith("FAST") else 200_000 if sims_mode.startswith("STD") else 600_000

    st.markdown("---")

    colA, colB = st.columns(2)
    with colA:
        do_run = st.button("🚀 Run pricing", type="primary")
    with colB:
        do_refresh = st.button("🔄 Refresh view")

    if st.button("🧹 Clear cache"):
        st.cache_data.clear()
        st.rerun()


# -------------------------------------------------
# Run pricing button
# -------------------------------------------------
if do_run:
    st.info(f"Running model: **{MODEL_MODULE_NAME}** | League: **{league}**")
    t0 = datetime.now()

    try:
        with st.spinner("Computing…"):
            run_model(
                league=league,
                margin=float(margin),
                over_price_boost=float(over_boost),
                alpha_squeeze=float(alpha_squeeze),
                n_sims=int(n_sims),
            )

        dt = (datetime.now() - t0).total_seconds()
        st.success(f"Done ✅ ({dt:.1f}s). Refreshing outputs…")
        st.cache_data.clear()
        st.rerun()

    except Exception as e:
        st.error("Model run failed ❌")
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        st.stop()


if do_refresh:
    st.rerun()


# -------------------------------------------------
# Load available outputs for selected league
# -------------------------------------------------
odds_files = list_files(f"{league}_bookmaker_odds_corners_gw*_margin*.csv")
teams_files = list_files(f"{league}_model_teams_table_gw*.csv")
sanity_files = list_files(f"{league}_model_sanity_table_gw*.csv")

if not odds_files:
    st.warning(
        f"Nema output fajlova za **{league}** u `{DOUT_DIR}`.\n\n"
        f"Klikni **Run pricing** ili proveri da li model zaista piše u `dout/`."
    )
    st.stop()

# Let user pick which odds file to view
with st.sidebar:
    st.subheader("View output file")
    odds_choice = st.selectbox(
        "Odds CSV (latest first)",
        odds_files,
        index=0,
        format_func=lambda p: os.path.basename(p),
    )

    teams_choice = teams_files[0] if teams_files else None
    sanity_choice = sanity_files[0] if sanity_files else None

    st.caption(f"Odds: `{os.path.basename(odds_choice)}`")
    if teams_choice:
        st.caption(f"Teams: `{os.path.basename(teams_choice)}`")
    if sanity_choice:
        st.caption(f"Sanity: `{os.path.basename(sanity_choice)}`")


@st.cache_data
def load_bundle(odds_path: str, teams_path: str | None, sanity_path: str | None):
    df_odds = load_csv(odds_path)
    df_teams = load_csv(teams_path) if teams_path else None
    df_sanity = load_csv(sanity_path) if sanity_path else None
    return df_odds, df_teams, df_sanity


df_odds, df_teams, df_sanity = load_bundle(odds_choice, teams_choice, sanity_choice)

# -------------------------------------------------
# Top metrics
# -------------------------------------------------
st.subheader(f"{league} — Pricing view")
c1, c2, c3, c4 = st.columns(4)

if df_sanity is not None:
    mu_league = safe_metric(df_sanity, "mu_league")
    alpha = safe_metric(df_sanity, "alpha")
    emp_mean = safe_metric(df_sanity, "emp_total_corners_mean")

    if mu_league is not None:
        c1.metric("μ league", f"{mu_league:.3f}")
    else:
        c1.metric("μ league", "n/a")

    if alpha is not None:
        c2.metric("alpha", f"{alpha:.4f}")
    else:
        c2.metric("alpha", "n/a")

    if emp_mean is not None:
        c3.metric("emp mean", f"{emp_mean:.3f}")
    else:
        c3.metric("emp mean", "n/a")
else:
    c1.metric("μ league", "n/a")
    c2.metric("alpha", "n/a")
    c3.metric("emp mean", "n/a")

# fixture count
try:
    fixtures = df_odds[["home_team", "away_team"]].drop_duplicates()
    c4.metric("fixtures priced", f"{len(fixtures)}")
except Exception:
    c4.metric("fixtures priced", f"{len(df_odds)}")


# -------------------------------------------------
# Tables / Tabs
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📌 Main lines", "📄 Full odds", "👥 Teams"])

with tab1:
    st.markdown("### Main lines (closest to 50/50)")

    if "is_main_line" in df_odds.columns:
        main_df = df_odds[df_odds["is_main_line"] == True].copy()
    else:
        # fallback: pick min absdiff per match if column missing
        if "p_over_absdiff_5050" in df_odds.columns:
            main_df = (
                df_odds.sort_values("p_over_absdiff_5050")
                .groupby(["home_team", "away_team"], as_index=False)
                .head(1)
            )
        else:
            main_df = df_odds.copy()

    cols = [c for c in ["home_team", "away_team", "main_line", "line", "p_over", "bookmaker_odds_over", "bookmaker_odds_under", "mu_match"] if c in main_df.columns]
    show = main_df[cols].copy()

    # normalize: show "Line" column
    if "main_line" in show.columns:
        show = show.rename(columns={"main_line": "Line"})
    elif "line" in show.columns:
        show = show.rename(columns={"line": "Line"})

    st.dataframe(show.sort_values("mu_match", ascending=False) if "mu_match" in show.columns else show, use_container_width=True)

    st.download_button(
        "⬇️ Download main lines CSV",
        show.to_csv(index=False).encode("utf-8"),
        file_name=f"{league}_main_lines.csv",
        mime="text/csv",
    )

with tab2:
    st.markdown("### Full odds table")
    st.dataframe(df_odds, use_container_width=True, height=650)

    st.download_button(
        "⬇️ Download full odds CSV",
        df_odds.to_csv(index=False).encode("utf-8"),
        file_name=os.path.basename(odds_choice),
        mime="text/csv",
    )

with tab3:
    st.markdown("### Team parameters")
    if df_teams is None:
        st.info("Nema teams table za ovaj league output.")
    else:
        st.dataframe(df_teams.sort_values("attack", ascending=False) if "attack" in df_teams.columns else df_teams,
                     use_container_width=True, height=650)

        st.download_button(
            "⬇️ Download teams CSV",
            df_teams.to_csv(index=False).encode("utf-8"),
            file_name=os.path.basename(teams_choice) if teams_choice else f"{league}_teams.csv",
            mime="text/csv",
        )

with st.expander("🔎 Debug: files in dout/"):
    st.write("DOUT_DIR:", DOUT_DIR)
    st.write("Odds files:", [os.path.basename(x) for x in odds_files[:20]])
    st.write("Teams files:", [os.path.basename(x) for x in teams_files[:20]])
    st.write("Sanity files:", [os.path.basename(x) for x in sanity_files[:20]])
