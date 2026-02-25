import streamlit as st

# IMPORTANT: set_page_config MUST be called before any other Streamlit commands
st.set_page_config(page_title="Corners Pricing Dashboard (Multi-league)", layout="wide")

import pandas as pd
import os
import glob
import sys
import hashlib
from datetime import datetime

# Allow importing local modules from repo root
sys.path.append(os.getcwd())

# ====================== MODEL IMPORT ======================
MODEL_MODULE_CANDIDATES = [
    "bookmaker_kornerv12_multileague_v4",
    # fallbacks if you rename the engine later
    "bookmaker_kornerv12_multileague",
]

model = None
_model_import_errs = []
for _m in MODEL_MODULE_CANDIDATES:
    try:
        model = __import__(_m)
        MODEL_MODULE_NAME = _m
        break
    except Exception as e:
        _model_import_errs.append(f"{_m}: {e}")

if model is None:
    st.error("❌ Ne mogu da importujem multileague engine. Pokušao sam:\n\n- " + "\n- ".join(_model_import_errs))
    st.stop()

# ====================== PASSWORD ZAŠTITA ======================
# NOTE: promijeni lozinku pre deploy-a (nikad ne drži pravu lozinku u repo-u)
PASSWORD = "tvoja_lozinka_2026"  # ← PROMENI OVO U NEŠTO JAKO!

def check_password_callback():
    if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == hashlib.sha256(PASSWORD.encode()).hexdigest():
        st.session_state.password_correct = True
        st.rerun()
    else:
        st.error("❌ Pogrešna lozinka")

def check_password():
    """Vrati True ako je password tačan."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.text_input(
            "Unesi lozinku za pristup",
            type="password",
            key="password",
            on_change=check_password_callback,
        )
        return False
    return True

if not check_password():
    st.stop()

# ====================== HELPERS ======================
OUT_DIR = getattr(model, "DOUT_DIR", "dout")

def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def _parse_gw_from_filename(path: str):
    base = os.path.basename(path)
    # ..._gw28_...
    try:
        part = base.split("_gw", 1)[1]
        gw = int(part.split("_", 1)[0].split(".", 1)[0])
        return gw
    except Exception:
        return None

def _list_files(pattern: str):
    files = glob.glob(os.path.join(OUT_DIR, pattern))
    files.sort(key=os.path.getmtime, reverse=True)
    return files

@st.cache_data(show_spinner=False)
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _fmt_ts(path: str):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

# ====================== UI ======================
st.title("⚽ Corners Pricing Dashboard — Multi-league")
st.caption(f"Engine: `{MODEL_MODULE_NAME}` | Output folder: `{OUT_DIR}`")

_ensure_out_dir()

leagues = getattr(model, "LEAGUES", ["EPL", "LL", "SA", "BL", "L1"])

with st.sidebar:
    st.header("Podešavanja")

    league = st.selectbox("Liga", leagues, index=0)

    st.subheader("Run parametri")
    margin = st.number_input("Margin", value=0.080, min_value=0.0, max_value=0.20, step=0.005, format="%.3f")
    over_price_boost = st.number_input("Over price boost", value=0.010, min_value=-0.10, max_value=0.10, step=0.005, format="%.3f")
    n_sims = st.number_input("n_sims", value=50_000, min_value=5_000, max_value=400_000, step=5_000)

    alpha_squeeze_txt = st.text_input("alpha_squeeze (prazno = None)", value="")
    alpha_squeeze = None
    if alpha_squeeze_txt.strip():
        try:
            alpha_squeeze = float(alpha_squeeze_txt.strip())
        except Exception:
            st.warning("alpha_squeeze mora biti broj ili prazno.")
            alpha_squeeze = None

    st.divider()
    run_btn = st.button("▶ Run model", type="primary")

# ====================== RUN ======================
if run_btn:
    with st.spinner(f"Running {league}..."):
        try:
            model.main(
                league=str(league),
                margin=float(margin),
                over_price_boost=float(over_price_boost),
                alpha_squeeze=alpha_squeeze,
                n_sims=int(n_sims),
            )
            st.success(f"✅ Završeno. Fajlovi su u: {OUT_DIR}")
        except Exception as e:
            st.error(f"❌ Model run failed: {e}")
            st.stop()

# ====================== TABS ======================
tab1, tab2, tab3 = st.tabs(["📊 Existing GW", "🧩 Match lookup", "🔬 Team Diagnostics"])

# -------- TAB 1: Existing GW (odds outputs) --------
with tab1:
    st.subheader(f"{league} — Odds output")

    odds_files = _list_files(f"{league}_bookmaker_odds_corners_gw*_margin*.csv")
    if not odds_files:
        st.info("Nema odds CSV fajlova za izabranu ligu. Klikni **Run model**.")
    else:
        # pick latest by mtime
        default = 0
        sel = st.selectbox(
            "Odaberi odds fajl",
            odds_files,
            index=default,
            format_func=lambda p: f"{os.path.basename(p)}  (mtime: {_fmt_ts(p)})",
        )
        df = _load_csv(sel)
        st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")

        # quick filters
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            team_q = st.text_input("Filter team (home/away sadrži)", "")
        with c2:
            cols = df.columns.tolist()
            sort_col = st.selectbox("Sort by", cols, index=cols.index("mu_total") if "mu_total" in cols else 0)
        with c3:
            asc = st.checkbox("Ascending", value=False)

        view = df.copy()
        if team_q.strip():
            q = team_q.strip().lower()
            hcol = "home_team" if "home_team" in view.columns else None
            acol = "away_team" if "away_team" in view.columns else None
            if hcol and acol:
                view = view[
                    view[hcol].astype(str).str.lower().str.contains(q)
                    | view[acol].astype(str).str.lower().str.contains(q)
                ]

        if sort_col in view.columns:
            view = view.sort_values(sort_col, ascending=asc)

        st.dataframe(view, use_container_width=True, height=620)

# -------- TAB 2: Match lookup (from existing CSV) --------
with tab2:
    st.subheader("Match lookup (bez re-fitovanja)")
    st.caption("Ovaj tab samo traži meč u već generisanom odds CSV-u (ne pokreće model).")

    odds_files = _list_files(f"{league}_bookmaker_odds_corners_gw*_margin*.csv")
    if not odds_files:
        st.info("Nema odds CSV fajlova. Prvo pokreni **Run model**.")
    else:
        sel = st.selectbox(
            "Odaberi odds fajl (iz kog tražimo meč)",
            odds_files,
            index=0,
            format_func=lambda p: os.path.basename(p),
            key="lookup_odds_file",
        )
        df = _load_csv(sel)

        home = st.text_input("Home (npr. Arsenal)", "")
        away = st.text_input("Away (npr. Chelsea)", "")

        if st.button("🔎 Find in file"):
            view = df.copy()
            if "home_team" in view.columns and home.strip():
                view = view[view["home_team"].astype(str).str.contains(home.strip(), case=False, na=False)]
            if "away_team" in view.columns and away.strip():
                view = view[view["away_team"].astype(str).str.contains(away.strip(), case=False, na=False)]
            if len(view) == 0:
                st.warning("Nije pronađen meč u ovom fajlu.")
            else:
                # show just the match rows
                st.dataframe(view, use_container_width=True, height=520)

# -------- TAB 3: Team Diagnostics (extended team table) --------
with tab3:
    st.subheader(f"{league} — Team table (extended)")

    team_files = _list_files(f"{league}_model_teams_table_gw*.csv")
    if not team_files:
        st.info("Nema team table fajlova za izabranu ligu. Klikni **Run model**.")
    else:
        sel = st.selectbox(
            "Odaberi team table fajl",
            team_files,
            index=0,
            format_func=lambda p: f"{os.path.basename(p)}  (mtime: {_fmt_ts(p)})",
        )
        df = _load_csv(sel)
        st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")

        # search
        q = st.text_input("Search team", "")
        view = df.copy()
        if q.strip() and "team" in view.columns:
            view = view[view["team"].astype(str).str.lower().str.contains(q.strip().lower())]

        # preferred column order (works even if some are missing)
        preferred = [
            "team",
            "attack", "defense",
            "matches_w",
            "mu_for_avg", "mu_against_avg", "mu_total_team_avg",
            "total_emp_mean", "total_emp_var",
            "for_emp_home", "for_emp_away",
            "against_emp_home", "against_emp_away",
            "mu_for_pred_home", "mu_for_pred_away",
            "mu_against_pred_home", "mu_against_pred_away",
            "emp_against_factor",
            "tempo_factor",
        ]
        cols = [c for c in preferred if c in view.columns] + [c for c in view.columns if c not in preferred]

        st.dataframe(view[cols], use_container_width=True, height=520)

        st.markdown("### Quick insights")
        c1, c2, c3 = st.columns(3)

        if "total_emp_var" in df.columns:
            with c1:
                st.write("Most volatile (emp var)")
                st.dataframe(df.sort_values("total_emp_var", ascending=False).head(8)[["team", "total_emp_var"]], use_container_width=True, height=260)
            with c2:
                st.write("Most stable (emp var)")
                st.dataframe(df.sort_values("total_emp_var", ascending=True).head(8)[["team", "total_emp_var"]], use_container_width=True, height=260)

        if "for_emp_home" in df.columns and "for_emp_away" in df.columns:
            tmp = df.copy()
            tmp["home_away_for_bias"] = tmp["for_emp_home"] - tmp["for_emp_away"]
            with c3:
                st.write("Home–Away 'for' bias (emp)")
                st.dataframe(
                    pd.concat(
                        [
                            tmp.sort_values("home_away_for_bias", ascending=False).head(6)[["team", "home_away_for_bias"]],
                            tmp.sort_values("home_away_for_bias", ascending=True).head(6)[["team", "home_away_for_bias"]],
                        ],
                        ignore_index=True,
                    ),
                    use_container_width=True,
                    height=260,
                )

st.divider()
st.caption("Tip: Ako želiš 'live overrides' (emp/tempo/hybrid) kroz UI bez editovanja engine-a, dodamo u engine `main_with_cfg(cfg, ...)` pa app šalje cfg direktno.")
