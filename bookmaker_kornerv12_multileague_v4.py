import os
import glob
import math
import logging
import zlib
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# =============================================================================
# CONFIG
# =============================================================================

# --- Core features ---
USE_EMP_BIAS_CORRECTION = True
EMP_BIAS_K = 25.0
EMP_BIAS_MAX_FACTOR = 1.30

USE_TEMPO_LAYER = False
TEMPO_K = 20.0
TEMPO_MAX_FACTOR = 1.25

USE_HYBRID_INTERACTION = True
HYBRID_W = 0.62

# --- Model control ---
ALPHA_MIN, ALPHA_MAX = 0.001, 0.03
ALPHA_SQUEEZE_BY_LEAGUE = {
    "EPL": 0.68,
    "BL":  0.72,
    "LL":  0.74,
    "L1":  0.75,
    "SA":  0.78,
}
ALPHA_SQUEEZE_DEFAULT = 0.68          # možeš menjati u main()

USE_TEAM_EFFECT_CAP = True
TEAM_EFFECT_CAP = 0.55

MU_LEAGUE_ANCHOR_LOW = 0.88
MU_LEAGUE_ANCHOR_HIGH = 1.12

SOFT_ANCHOR_W_BASE = 0.08
SOFT_ANCHOR_W_UPWARD = 0.18

# --- Mismatch inflate ---
MISMATCH_INFLATE_ENABLED = True
MISMATCH_GAP_THRESHOLD = 0.30
MISMATCH_INFLATE_SLOPE = 0.08
MISMATCH_INFLATE_CAP = 0.15

# --- Chaos release (league-controlled cap) ---
CHAOS_DEF_NEAR0_BAND = 0.10   # abs(def) < 0.10 -> "non-control defending"
CHAOS_ATTACK_THRESH = -0.20   # low-possession trigger
CHAOS_MAX = 0.05              # EPL default; other leagues override via preset

# --- IO / folders ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")   # samo input fajlovi
DOUT_DIR = os.path.join(BASE_DIR, "dout")   # svi rezultati

LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

ACTIVE_SEASON_HINT = "2025-to-2026"

# FootyStats columns
COL_HOME_TEAM = "home_team_name"
COL_AWAY_TEAM = "away_team_name"
COL_HOME_CORNERS = "home_team_corner_count"
COL_AWAY_CORNERS = "away_team_corner_count"
COL_STATUS = "status"
COL_GAMEWEEK = "Game Week"

COL_HOME_GOALS = "home_team_goal_count"
COL_AWAY_GOALS = "away_team_goal_count"

COMPLETED_STATUSES = {"complete", "finished", "ft", "ended"}

DEFAULT_LINES = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]

# =============================================================================
# LEAGUE SUPPORT (5 leagues)
# =============================================================================

LEAGUES = ["EPL", "LL", "SA", "BL", "L1"]

MATCH_PATTERNS = {
    # EPL / Premier League
    "EPL": [
        "*premier-league-matches*-stats*.csv",
        "*england-premier-league-matches*-stats*.csv",
    ],
    # La Liga
    "LL": [
        "*la-liga-matches*-stats*.csv",
        "*spain-la-liga-matches*-stats*.csv",
    ],
    # Serie A
    "SA": [
        "*serie-a-matches*-stats*.csv",
        "*italy-serie-a-matches*-stats*.csv",
    ],
    # Bundesliga
    "BL": [
        "*bundesliga-matches*-stats*.csv",
        "*germany-bundesliga-matches*-stats*.csv",
    ],
    # Ligue 1
    "L1": [
        "*ligue-1-matches*-stats*.csv",
        "*france-ligue-1-matches*-stats*.csv",
        "*french-ligue-1-matches*-stats*.csv",
    ],
}

# Season weights per league (start conservative; adjust as you gather more history)
SEASON_WEIGHTS_BY_LEAGUE = {
    "EPL": {"2023-to-2024": 0.0, "2024-to-2025": 1.0, "2025-to-2026": 1.0},
    "LL":  {"2023-to-2024": 0.0, "2024-to-2025": 1.0, "2025-to-2026": 1.0},
    "SA":  {"2023-to-2024": 0.0, "2024-to-2025": 1.0, "2025-to-2026": 1.0},
    "BL":  {"2023-to-2024": 0.0, "2024-to-2025": 1.0, "2025-to-2026": 1.0},
    "L1":  {"2023-to-2024": 0.0, "2024-to-2025": 1.0, "2025-to-2026": 1.0},
}

# League-specific presets (calibration knobs)
LEAGUE_PRESETS = {
    "EPL": dict(
        EMP_BIAS_K=25.0,
        EMP_BIAS_MAX_FACTOR=1.30,
        SOFT_ANCHOR_W_BASE=0.08,
        SOFT_ANCHOR_W_UPWARD=0.18,
        CHAOS_MAX=0.05,
    ),
    "LL": dict(
        EMP_BIAS_K=30.0,
        EMP_BIAS_MAX_FACTOR=1.25,
        SOFT_ANCHOR_W_BASE=0.07,
        SOFT_ANCHOR_W_UPWARD=0.14,
        CHAOS_MAX=0.020,
    ),
    "SA": dict(
        EMP_BIAS_K=36.0,
        EMP_BIAS_MAX_FACTOR=1.25,
        SOFT_ANCHOR_W_BASE=0.055,
        SOFT_ANCHOR_W_UPWARD=0.12,
        CHAOS_MAX=0.018,
    ),
    "BL": dict(
        EMP_BIAS_K=27.0,
        EMP_BIAS_MAX_FACTOR=1.28,
        SOFT_ANCHOR_W_BASE=0.07,
        SOFT_ANCHOR_W_UPWARD=0.16,
        CHAOS_MAX=0.025,
    ),
    "L1": dict(
        EMP_BIAS_K=34.0,
        EMP_BIAS_MAX_FACTOR=1.25,
        SOFT_ANCHOR_W_BASE=0.06,
        SOFT_ANCHOR_W_UPWARD=0.14,
        CHAOS_MAX=0.020,
    ),
}


# =============================================================================
# LEAGUE-SCOPED CONFIG (no mutable globals)
# =============================================================================

@dataclass(frozen=True)
class LeagueConfig:
    league: str
    # runtime / tuning knobs (safe to tweak from app.py)
    use_emp_bias_correction: bool = True
    emp_bias_k: float = 25.0
    emp_bias_max_factor: float = 1.30

    use_tempo_layer: bool = False
    tempo_k: float = 20.0
    tempo_max_factor: float = 1.25

    use_hybrid_interaction: bool = True
    hybrid_w: float = 0.62

    use_team_effect_cap: bool = True
    team_effect_cap: float = 0.55

    # league anchoring
    mu_league_anchor_low: float = 0.88
    mu_league_anchor_high: float = 1.12
    soft_anchor_w_base: float = 0.08
    soft_anchor_w_upward: float = 0.18

    # mismatch inflate
    mismatch_inflate_enabled: bool = True
    mismatch_gap_threshold: float = 0.30
    mismatch_inflate_slope: float = 0.08
    mismatch_inflate_cap: float = 0.15

    # chaos release
    chaos_def_near0_band: float = 0.10
    chaos_attack_thresh: float = -0.20
    chaos_max: float = 0.05

    # numerical safety (used by empirical/tempo ratios)
    ratio_clip_low: float = 0.50
    ratio_clip_high: float = 2.00

    # distribution control
    alpha_squeeze: float = 0.68


def get_league_config(
    league: str,
    log: logging.Logger,
    *,
    alpha_squeeze_override: Optional[float] = None,
    overrides: Optional[Dict[str, float]] = None,
) -> LeagueConfig:
    """Build a per-league immutable config.

    - Starts from module defaults (the constants above)
    - Applies LEAGUE_PRESETS overrides (if present)
    - Applies alpha_squeeze per league (or explicit override)
    - Applies explicit overrides dict last (for app.py sliders)
    """
    base = LeagueConfig(
        league=league,
        use_emp_bias_correction=bool(USE_EMP_BIAS_CORRECTION),
        emp_bias_k=float(EMP_BIAS_K),
        emp_bias_max_factor=float(EMP_BIAS_MAX_FACTOR),
        use_tempo_layer=bool(USE_TEMPO_LAYER),
        tempo_k=float(TEMPO_K),
        tempo_max_factor=float(TEMPO_MAX_FACTOR),
        use_hybrid_interaction=bool(USE_HYBRID_INTERACTION),
        hybrid_w=float(HYBRID_W),
        use_team_effect_cap=bool(USE_TEAM_EFFECT_CAP),
        team_effect_cap=float(TEAM_EFFECT_CAP),
        mu_league_anchor_low=float(MU_LEAGUE_ANCHOR_LOW),
        mu_league_anchor_high=float(MU_LEAGUE_ANCHOR_HIGH),
        soft_anchor_w_base=float(SOFT_ANCHOR_W_BASE),
        soft_anchor_w_upward=float(SOFT_ANCHOR_W_UPWARD),
        mismatch_inflate_enabled=bool(MISMATCH_INFLATE_ENABLED),
        mismatch_gap_threshold=float(MISMATCH_GAP_THRESHOLD),
        mismatch_inflate_slope=float(MISMATCH_INFLATE_SLOPE),
        mismatch_inflate_cap=float(MISMATCH_INFLATE_CAP),
        chaos_def_near0_band=float(CHAOS_DEF_NEAR0_BAND),
        chaos_attack_thresh=float(CHAOS_ATTACK_THRESH),
        chaos_max=float(CHAOS_MAX),
        ratio_clip_low=0.50,
        ratio_clip_high=2.00,
        alpha_squeeze=float(ALPHA_SQUEEZE_BY_LEAGUE.get(league, ALPHA_SQUEEZE_DEFAULT)),
    )

    # apply league preset if present (do NOT mutate globals)
    if league in LEAGUE_PRESETS:
        p = LEAGUE_PRESETS[league]
        base = dataclass_replace(
            base,
            emp_bias_k=float(p.get("EMP_BIAS_K", base.emp_bias_k)),
            emp_bias_max_factor=float(p.get("EMP_BIAS_MAX_FACTOR", base.emp_bias_max_factor)),
            soft_anchor_w_base=float(p.get("SOFT_ANCHOR_W_BASE", base.soft_anchor_w_base)),
            soft_anchor_w_upward=float(p.get("SOFT_ANCHOR_W_UPWARD", base.soft_anchor_w_upward)),
            chaos_max=float(p.get("CHAOS_MAX", base.chaos_max)),
        )
        log.info(
            "Applied preset %s: EMP_BIAS_K=%.1f, EMP_BIAS_MAX_FACTOR=%.2f, SOFT_ANCHOR_W_BASE=%.3f, SOFT_ANCHOR_W_UPWARD=%.3f, CHAOS_MAX=%.3f",
            league, base.emp_bias_k, base.emp_bias_max_factor, base.soft_anchor_w_base, base.soft_anchor_w_upward, base.chaos_max
        )
    else:
        log.warning("No league preset for %s; using default config values.", league)

    # alpha squeeze override (explicit argument wins)
    if alpha_squeeze_override is not None:
        base = dataclass_replace(base, alpha_squeeze=float(alpha_squeeze_override))

    # explicit overrides last (from app.py sliders etc.)
    if overrides:
        safe = {k: v for k, v in overrides.items() if hasattr(base, k)}
        if safe:
            base = dataclass_replace(base, **safe)

    return base


def dataclass_replace(dc_obj: LeagueConfig, **changes) -> LeagueConfig:
    # local helper to avoid importing dataclasses.replace in multiple places
    return dataclasses.replace(dc_obj, **changes)


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(tag: str = "") -> logging.Logger:
    log = logging.getLogger("bookmaker")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fname = f"bookmaker_{tag}_{ts}.txt" if tag else f"bookmaker_{ts}.txt"
    fpath = os.path.join(LOG_DIR, fname)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(fpath, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    log.info("Logging started → %s", fpath)
    return log


# =============================================================================
# BOOKMAKER ODDS
# =============================================================================

def bookmaker_odds_two_way(p_over: float, margin: float, over_price_boost: float = 0.0):
    p_over = float(np.clip(p_over, 1e-9, 1.0 - 1e-9))
    p_under = 1.0 - p_over
    target = 1.0 + float(margin)

    imp_over = p_over * target
    imp_under = p_under * target

    if over_price_boost != 0.0:
        shift = float(over_price_boost) * imp_over
        imp_over = max(1e-9, imp_over - shift)
        imp_under = max(1e-9, imp_under + shift)

    s = imp_over + imp_under
    if s > 0:
        scale = target / s
        imp_over *= scale
        imp_under *= scale

    return 1.0 / imp_over, 1.0 / imp_under, imp_over, imp_under


# =============================================================================
# HELPERS
# =============================================================================
# -----------------------------------------------------------------------------
# Team name normalization
# -----------------------------------------------------------------------------
# IMPORTANT:
# - We must use the SAME normalization for: training, pricing, and any diagnostics
# - La Liga (Spain) often has accents and naming variants -> we use accent stripping
#   + league-specific aliases.
import re
import unicodedata

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def _basic_norm(s: str) -> str:
    s = _strip_accents(str(s).strip())
    s = s.lower()
    # keep alnum + spaces only
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _titleish_from_key(key: str) -> str:
    # Pretty-but-stable canonical form derived from normalized key.
    # Keep common particles lower-case and standardize common acronyms.
    lower_words = {"de", "del", "la", "las", "los", "y", "da", "do", "di", "of"}

    # Acronyms that should stay upper-case in canonical names
    acronyms = {
        "fc": "FC",
        "cf": "CF",
        "sv": "SV",
        "sc": "SC",
        "tsg": "TSG",
        "rb": "RB",
        "ud": "UD",
        "rcd": "RCD",
        "cd": "CD",
        "rc": "RC",
        "vfb": "VfB",   # common German styling
        "vfl": "VfL",
        "bvb": "BVB",
    }

    parts = []
    for w in key.split():
        if w in lower_words:
            parts.append(w)
            continue
        if w in acronyms:
            parts.append(acronyms[w])
            continue
        parts.append(w[:1].upper() + w[1:])

    return " ".join(parts)

def _mk_alias_map(d: Dict[str, str]) -> Dict[str, str]:
    # Store alias keys in normalized form
    return {_basic_norm(k): v for k, v in d.items()}

COMMON_TEAM_ALIASES = _mk_alias_map({
    # EPL / common English variants
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham",
    "Nottingham Forest": "Nott'm Forest",
    "Newcastle United": "Newcastle",
})

LEAGUE_TEAM_ALIASES: Dict[str, Dict[str, str]] = {
    # La Liga: accents and frequent naming variants
    "LL": _mk_alias_map({
        "Atlético Madrid": "Atletico Madrid",
        "Deportivo Alavés": "Alaves",
        "Deportivo Alaves": "Alaves",
        "Alavés": "Alaves",
        "Celta de Vigo": "Celta Vigo",
        "Real Sociedad de Fútbol": "Real Sociedad",
        "Rayo Vallecano de Madrid": "Rayo Vallecano",
        "R.C.D. Mallorca": "Mallorca",
        "RCD Mallorca": "Mallorca",
        "Athletic Club": "Athletic Bilbao",
        "Athletic Club de Bilbao": "Athletic Bilbao",
        "Club Atlético Osasuna": "Osasuna",
        "Real Betis Balompié": "Real Betis",
        "Villarreal CF": "Villarreal",
        "Valencia CF": "Valencia",
        "Getafe CF": "Getafe",
        "Sevilla FC": "Sevilla",
        "FC Barcelona": "Barcelona",
        "Real Madrid CF": "Real Madrid",
        "Real Club Celta de Vigo": "Celta Vigo",
        "UD Las Palmas": "Las Palmas",
    }),

    # Bundesliga: umlauts, abbreviations, and frequent naming variants
    "BL": _mk_alias_map({
        "Bayern München": "Bayern Munchen",
        "Bayern Muenchen": "Bayern Munchen",
        "Bayern Munich": "Bayern Munchen",
        "FC Bayern München": "Bayern Munchen",
        "FC Bayern Munich": "Bayern Munchen",
        "Borussia M'gladbach": "Borussia Monchengladbach",
        "Borussia Mönchengladbach": "Borussia Monchengladbach",
        "Borussia Monchengladbach": "Borussia Monchengladbach",
        "Gladbach": "Borussia Monchengladbach",
        "1. FC Köln": "Koln",
        "1 FC Köln": "Koln",
        "1. FC Koln": "Koln",
        "FC Köln": "Koln",
        "FC Koln": "Koln",
        "Köln": "Koln",
        "TSG 1899 Hoffenheim": "Hoffenheim",
        "TSG Hoffenheim": "Hoffenheim",
        "1899 Hoffenheim": "Hoffenheim",
        "SC Freiburg": "Freiburg",
        "Sport-Club Freiburg": "Freiburg",
        "1. FSV Mainz 05": "Mainz 05",
        "FSV Mainz 05": "Mainz 05",
        "Mainz": "Mainz 05",
        "Bayer 04 Leverkusen": "Bayer Leverkusen",
        "Bayer Leverkusen": "Bayer Leverkusen",
        "FC Augsburg": "Augsburg",
        "VfB Stuttgart": "Stuttgart",
        "VFB Stuttgart": "Stuttgart",
        "FC Union Berlin": "Union Berlin",
        "1. FC Union Berlin": "Union Berlin",
        "SV Werder Bremen": "Werder Bremen",
        "Werder Bremen": "Werder Bremen",
        "VfL Wolfsburg": "Wolfsburg",
        "RB Leipzig": "RB Leipzig",
        "Hamburg": "Hamburger SV",
        "Hamburger SV": "Hamburger SV",
        "FC St. Pauli": "St Pauli",
        "St. Pauli": "St Pauli",
        "FC Heidenheim": "Heidenheim",
        "1. FC Heidenheim": "Heidenheim",
        "Eintracht Frankfurt": "Eintracht Frankfurt",
        "Borussia Dortmund": "Borussia Dortmund",
    }),
}

def normalize_team_name(name: str, league: Optional[str] = None) -> str:
    """Normalize team names consistently across the pipeline.

    - strips accents (important for Spain)
    - applies common aliases (cross-league)
    - applies league-specific aliases (if league is provided)
    - otherwise returns a stable canonical title-ish form derived from the normalized key
    """
    raw = str(name).strip()
    key = _basic_norm(raw)

    if league:
        amap = LEAGUE_TEAM_ALIASES.get(str(league), {})
        if key in amap:
            return amap[key]

    if key in COMMON_TEAM_ALIASES:
        return COMMON_TEAM_ALIASES[key]

    # Default: accentless, cleaned, stable formatting
    return _titleish_from_key(key)


def find_match_files(data_dir: str, patterns: List[str]) -> List[str]:
    out = []
    for pat in patterns:
        out.extend(glob.glob(os.path.join(data_dir, pat)))
    # unique + stable order
    out = sorted(list(set(out)))
    return out


def season_weight_from_filename(path: str, weights: Dict[str, float]) -> float:
    base = os.path.basename(path)
    for k, w in weights.items():
        if k in base:
            return float(w)
    # default if unknown
    return 1.0


def _as_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def load_matches(path: str, league: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure columns exist
    required = [COL_HOME_TEAM, COL_AWAY_TEAM, COL_HOME_CORNERS, COL_AWAY_CORNERS, COL_STATUS, COL_GAMEWEEK]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {path}")

    # Normalize types
    df[COL_HOME_CORNERS] = _as_float_series(df, COL_HOME_CORNERS)
    df[COL_AWAY_CORNERS] = _as_float_series(df, COL_AWAY_CORNERS)
    df[COL_GAMEWEEK] = pd.to_numeric(df[COL_GAMEWEEK], errors="coerce")
    df[COL_STATUS] = df[COL_STATUS].astype(str).str.lower().str.strip()

    # Treat negative corner counts (e.g. -1 sentinel) as missing
    df.loc[df[COL_HOME_CORNERS] < 0, COL_HOME_CORNERS] = np.nan
    df.loc[df[COL_AWAY_CORNERS] < 0, COL_AWAY_CORNERS] = np.nan

    # Normalize team names (MUST be consistent across training/pricing/diagnostics)
    df[COL_HOME_TEAM] = df[COL_HOME_TEAM].astype(str).map(lambda x: normalize_team_name(x, league))
    df[COL_AWAY_TEAM] = df[COL_AWAY_TEAM].astype(str).map(lambda x: normalize_team_name(x, league))

    return df



def is_completed(df: pd.DataFrame) -> pd.Series:
    return df[COL_STATUS].isin(COMPLETED_STATUSES)


def stable_seed_from_match(home: str, away: str, gw: int) -> int:
    s = f"{home}|{away}|{gw}"
    return zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF


# =============================================================================
# MODEL (Negative Binomial)
# =============================================================================

@dataclass
class FittedModel:
    teams: List[str]
    beta0: float
    home_adv: float
    alpha: float
    attack: Dict[str, float] = field(default_factory=dict)
    defense: Dict[str, float] = field(default_factory=dict)
    matches_w: Dict[str, float] = field(default_factory=dict)
    team_tempo_factor: Dict[str, float] = field(default_factory=dict)
    emp_against_factor: Dict[str, float] = field(default_factory=dict)
    mu_league: float = 0.0


def nb2_logpmf(k: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """Element-wise NB2 log-PMF.
    NB2 parameterization: Var = mu + alpha*mu^2
    Returns an array of the same shape as the broadcast of k and mu.
    """
    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # keep parameters in valid numerical range
    alpha = float(np.clip(alpha, 1e-12, 1e3))
    mu = np.clip(mu, 1e-12, 1e6)

    r = 1.0 / alpha
    p = r / (r + mu)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)

    return (
        gammaln(k + r)
        - gammaln(r)
        - gammaln(k + 1.0)
        + r * np.log(p)
        + k * np.log1p(-p)
    )


def nb2_loglik(k: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """Total NB2 log-likelihood (sum over observations)."""
    return float(np.sum(nb2_logpmf(k, mu, alpha)))


def fit_nb_team_model(df: pd.DataFrame, log: logging.Logger) -> FittedModel:
    # Build team list
    teams = sorted(list(set(df[COL_HOME_TEAM]).union(set(df[COL_AWAY_TEAM]))))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # Observations (only completed)
    d = df[is_completed(df)].copy()
    d = d.dropna(subset=[COL_HOME_CORNERS, COL_AWAY_CORNERS, COL_GAMEWEEK])
    d = d[(d[COL_HOME_CORNERS] >= 0) & (d[COL_AWAY_CORNERS] >= 0)]
    if len(d) == 0:
        raise ValueError("No completed matches found for training.")

    # Weights (season weights)
    w = d["season_weight"].astype(float).to_numpy()

    # Base y
    y_home = d[COL_HOME_CORNERS].astype(int).to_numpy()
    y_away = d[COL_AWAY_CORNERS].astype(int).to_numpy()

    home_i = d[COL_HOME_TEAM].map(idx).to_numpy()
    away_i = d[COL_AWAY_TEAM].map(idx).to_numpy()

    # Exposure per team (weighted count)
    matches_w = {t: 0.0 for t in teams}
    for hi, ai, wi in zip(home_i, away_i, w):
        matches_w[teams[hi]] += float(wi)
        matches_w[teams[ai]] += float(wi)

    # Regularization scale per team
    exposure = np.array([matches_w[t] for t in teams], dtype=float)
    lam_vec = L2_LAMBDA_TEAM_BASE * (EXPOSURE_EPS / (exposure + EXPOSURE_EPS))

    # Parameters: beta0, home_adv, attack[n], defense[n], log_alpha
    x0 = np.zeros(2 + 2 * n + 1, dtype=float)
    x0[0] = math.log(max(1e-6, np.mean(np.r_[y_home, y_away])))  # beta0 init in log space
    x0[1] = 0.10  # home_adv
    log_alpha0 = math.log(0.01)
    x0[-1] = log_alpha0

    def unpack(x: np.ndarray):
        beta0 = x[0]
        home_adv = x[1]
        attack = x[2:2 + n]
        defense = x[2 + n:2 + 2 * n]
        alpha = float(np.exp(x[-1]))
        return beta0, home_adv, attack, defense, alpha

    def clip_team_effects(arr: np.ndarray) -> np.ndarray:
        if not USE_TEAM_EFFECT_CAP:
            return arr
        return np.clip(arr, -TEAM_EFFECT_CAP, TEAM_EFFECT_CAP)

    def neg_loglike(x: np.ndarray) -> float:
        beta0, home_adv, a, dfn, alpha = unpack(x)

        a = clip_team_effects(a)
        dfn = clip_team_effects(dfn)

        lin_h = beta0 + home_adv + a[home_i] + dfn[away_i]
        lin_a = beta0 + a[away_i] + dfn[home_i]
        # prevent overflow/underflow in exp
        mu_h = np.exp(np.clip(lin_h, -20.0, 20.0))
        mu_a = np.exp(np.clip(lin_a, -20.0, 20.0))

        # Weighted loglik
        ll = 0.0
        ll += float(np.sum(w * nb2_logpmf(y_home, mu_h, alpha)))
        ll += float(np.sum(w * nb2_logpmf(y_away, mu_a, alpha)))

        # Regularization
        # L2 on team effects with per-team lambda
        reg = 0.0
        reg += np.sum(lam_vec * (a ** 2))
        reg += np.sum(lam_vec * (dfn ** 2))
        reg += L2_LAMBDA_HOMEADV * (home_adv ** 2)

        # alpha clamp penalty
        if alpha < ALPHA_MIN:
            reg += (ALPHA_MIN - alpha) * 1000.0
        if alpha > ALPHA_MAX:
            reg += (alpha - ALPHA_MAX) * 1000.0
        if not np.isfinite(ll) or not np.isfinite(reg):
            return 1e18
        return -(ll) + reg

    res = minimize(neg_loglike, x0, method="L-BFGS-B")
    if not res.success:
        log.warning("Optimization not fully successful: %s", res.message)

    beta0, home_adv, a, dfn, alpha = unpack(res.x)
    a = np.clip(a, -TEAM_EFFECT_CAP, TEAM_EFFECT_CAP) if USE_TEAM_EFFECT_CAP else a
    dfn = np.clip(dfn, -TEAM_EFFECT_CAP, TEAM_EFFECT_CAP) if USE_TEAM_EFFECT_CAP else dfn
    alpha = float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))

    model = FittedModel(
        teams=teams,
        beta0=float(beta0),
        home_adv=float(home_adv),
        alpha=float(alpha),
        attack={t: float(a[idx[t]]) for t in teams},
        defense={t: float(dfn[idx[t]]) for t in teams},
        matches_w=matches_w,
        team_tempo_factor={t: 1.0 for t in teams},
        emp_against_factor={t: 1.0 for t in teams},
        mu_league=0.0,
    )

    return model


# =============================================================================
# EMP BIAS + TEMPO (helpers)
# =============================================================================

def compute_emp_against_factor(model: FittedModel, df_train: pd.DataFrame, cfg: LeagueConfig, log: logging.Logger) -> Dict[str, float]:
    # Empirical conceded corners vs model predicted conceded corners
    d = df_train[is_completed(df_train)].copy()
    d = d.dropna(subset=[COL_HOME_CORNERS, COL_AWAY_CORNERS])
    d = d[(d[COL_HOME_CORNERS] >= 0) & (d[COL_AWAY_CORNERS] >= 0)]
    d = d[(d[COL_HOME_CORNERS] >= 0) & (d[COL_AWAY_CORNERS] >= 0)]
    if len(d) == 0:
        return {t: 1.0 for t in model.teams}

    conceded_emp = {t: 0.0 for t in model.teams}
    conceded_pred = {t: 0.0 for t in model.teams}
    counts = {t: 0.0 for t in model.teams}

    for _, r in d.iterrows():
        h = r[COL_HOME_TEAM]
        a = r[COL_AWAY_TEAM]
        y_h = float(r[COL_HOME_CORNERS])
        y_a = float(r[COL_AWAY_CORNERS])
        w = float(r.get("season_weight", 1.0))

        # conceded: home concedes away corners; away concedes home corners
        conceded_emp[h] += w * y_a
        conceded_emp[a] += w * y_h

        mu_h = math.exp(model.beta0 + model.home_adv + model.attack.get(h, 0.0) + model.defense.get(a, 0.0))
        mu_a = math.exp(model.beta0 + model.attack.get(a, 0.0) + model.defense.get(h, 0.0))

        conceded_pred[h] += w * mu_a
        conceded_pred[a] += w * mu_h

        counts[h] += w
        counts[a] += w

    fac = {}
    for t in model.teams:
        if counts[t] <= 0:
            fac[t] = 1.0
            continue
        emp = conceded_emp[t] / counts[t]
        pred = conceded_pred[t] / counts[t]
        raw = (emp + 1e-9) / (pred + 1e-9)
        if not np.isfinite(raw):
            raw = 1.0
        raw = float(np.clip(raw, cfg.ratio_clip_low, cfg.ratio_clip_high))

        # shrink toward 1.0 using cfg.emp_bias_k
        shrink = counts[t] / (counts[t] + float(cfg.emp_bias_k))
        f = 1.0 + shrink * (raw - 1.0)

        # clamp
        f = float(np.clip(f, 1.0 / cfg.emp_bias_max_factor, cfg.emp_bias_max_factor))
        fac[t] = f

    log.info("Emp-bias: mean factor=%.3f, min=%.3f, max=%.3f",
             float(np.mean(list(fac.values()))),
             float(np.min(list(fac.values()))),
             float(np.max(list(fac.values()))))
    return fac


def compute_team_tempo_factor(model: FittedModel, df_train: pd.DataFrame, cfg: LeagueConfig, log: logging.Logger) -> Dict[str, float]:
    # Optional: team tempo based on total corners relative to league
    d = df_train[is_completed(df_train)].copy()
    d = d.dropna(subset=[COL_HOME_CORNERS, COL_AWAY_CORNERS])
    if len(d) == 0:
        return {t: 1.0 for t in model.teams}

    totals = (d[COL_HOME_CORNERS] + d[COL_AWAY_CORNERS]).astype(float)
    league_mean = float(totals.mean()) if len(totals) else 1.0

    emp_team = {t: 0.0 for t in model.teams}
    cnt_team = {t: 0.0 for t in model.teams}

    for _, r in d.iterrows():
        h = r[COL_HOME_TEAM]
        a = r[COL_AWAY_TEAM]
        total = float(r[COL_HOME_CORNERS] + r[COL_AWAY_CORNERS])
        w = float(r.get("season_weight", 1.0))

        emp_team[h] += w * total
        emp_team[a] += w * total
        cnt_team[h] += w
        cnt_team[a] += w

    fac = {}
    for t in model.teams:
        if cnt_team[t] <= 0:
            fac[t] = 1.0
            continue
        emp = emp_team[t] / cnt_team[t]
        raw = (emp + 1e-9) / (league_mean + 1e-9)
        if not np.isfinite(raw):
            raw = 1.0
        raw = float(np.clip(raw, cfg.ratio_clip_low, cfg.ratio_clip_high))

        shrink = cnt_team[t] / (cnt_team[t] + float(cfg.tempo_k))
        f = 1.0 + shrink * (raw - 1.0)
        f = float(np.clip(f, 1.0 / cfg.tempo_max_factor, cfg.tempo_max_factor))
        fac[t] = f

    log.info("Tempo: mean factor=%.3f, min=%.3f, max=%.3f",
             float(np.mean(list(fac.values()))),
             float(np.min(list(fac.values()))),
             float(np.max(list(fac.values()))))
    return fac


# =============================================================================
# MATCH μ + SIMULATION
# =============================================================================

def alpha_for_match(base_alpha: float, alpha_squeeze: float) -> float:
    # squeeze alpha toward ALPHA_MIN to reduce tails if desired
    a = float(base_alpha)
    a = ALPHA_MIN + (a - ALPHA_MIN) * float(np.clip(alpha_squeeze, 0.0, 1.0))
    return float(np.clip(a, ALPHA_MIN, ALPHA_MAX))


def simulate_total_corners(
    model: FittedModel,
    cfg: LeagueConfig,
    home: str,
    away: str,
    mu_league: float,
    rng: np.random.Generator,
    n_sims: int = 50_000,
    return_debug: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
    """
    Simulate total corners for a fixture using NB2 (gamma-poisson mixture).

    If return_debug=True, also returns a dict of per-match intermediate constants
    that are useful for diagnostics / calibration and for exposing model internals
    in the output tables.
    """

    # ------------------------------------------------------------------
    # Base μ (raw)
    # ------------------------------------------------------------------
    mu_h0 = math.exp(model.beta0 + model.home_adv + model.attack.get(home, 0.0) + model.defense.get(away, 0.0))
    mu_a0 = math.exp(model.beta0 + model.attack.get(away, 0.0) + model.defense.get(home, 0.0))

    mu_h = mu_h0
    mu_a = mu_a0
    mu_total_raw = mu_h + mu_a

    # ------------------------------------------------------------------
    # Hybrid interaction (optional)
    # ------------------------------------------------------------------
    hybrid_used = int(cfg.use_hybrid_interaction)
    split0 = float(np.clip(mu_h / max(1e-9, mu_total_raw), 0.30, 0.70))

    if cfg.use_hybrid_interaction:
        mu_total = mu_h + mu_a
        # mild additive blending toward league mean shape
        mu_total_hybrid = cfg.hybrid_w * mu_total + (1.0 - cfg.hybrid_w) * (0.5 * mu_total + 0.5 * mu_league)
        # keep split ratio stable
        split = float(np.clip(mu_h / max(1e-9, mu_total), 0.30, 0.70))
        mu_h = mu_total_hybrid * split
        mu_a = mu_total_hybrid * (1.0 - split)

    mu_total_after_hybrid = mu_h + mu_a

    # ------------------------------------------------------------------
    # Emp-bias correction (optional)
    # ------------------------------------------------------------------
    emp_used = int(cfg.use_emp_bias_correction and bool(model.emp_against_factor))
    emp_fac_home = float(model.emp_against_factor.get(away, 1.0)) if emp_used else 1.0
    emp_fac_away = float(model.emp_against_factor.get(home, 1.0)) if emp_used else 1.0
    if emp_used:
        mu_h *= emp_fac_home
        mu_a *= emp_fac_away

    mu_total_after_emp = mu_h + mu_a

    # ------------------------------------------------------------------
    # Mismatch inflation (optional)
    # ------------------------------------------------------------------
    mismatch_used = int(cfg.mismatch_inflate_enabled)
    gap = abs(float(model.attack.get(home, 0.0)) - float(model.attack.get(away, 0.0)))
    infl = 1.0
    if cfg.mismatch_inflate_enabled and gap > cfg.mismatch_gap_threshold:
        infl = 1.0 + min(cfg.mismatch_inflate_cap, cfg.mismatch_inflate_slope * (gap - cfg.mismatch_gap_threshold))
        mu_h *= infl
        mu_a *= infl

    mu_total_after_mismatch = mu_h + mu_a

    # ------------------------------------------------------------------
    # CHAOS RELEASE (league-controlled)
    # Corrects NB bias where low-possession teams look like low-event.
    # ------------------------------------------------------------------
    away_att = float(model.attack.get(away, 0.0))
    away_def = float(model.defense.get(away, 0.0))
    home_att = float(model.attack.get(home, 0.0))
    home_def = float(model.defense.get(home, 0.0))

    chaos_factor = 1.00

    # Soft chaos scaling (0..cfg.chaos_max) depending on how "control-like" defending looks.
    # Trigger: low-possession / low-attack team AND near-zero defending parameter
    def _chaos_scale(att: float, deff: float) -> float:
        if att > cfg.chaos_attack_thresh:
            return 0.0
        if abs(deff) > cfg.chaos_def_near0_band:
            return 0.0
        # scale up as def approaches 0 and att is lower
        s_def = 1.0 - min(1.0, abs(deff) / max(1e-9, cfg.chaos_def_near0_band))
        s_att = min(1.0, abs(att - cfg.chaos_attack_thresh) / 0.40)  # 0.40 is a soft range
        return float(np.clip(s_def * s_att, 0.0, 1.0))

    s_home = _chaos_scale(home_att, home_def)
    s_away = _chaos_scale(away_att, away_def)

    # apply to the "low-control" team only, but cap globally
    chaos_add = float(np.clip(max(s_home, s_away) * cfg.chaos_max, 0.0, cfg.chaos_max))
    if chaos_add > 0:
        chaos_factor = 1.0 + chaos_add
        mu_h *= chaos_factor
        mu_a *= chaos_factor

    mu_total_after_chaos = mu_h + mu_a

    # ------------------------------------------------------------------
    # Tempo layer (optional)
    # ------------------------------------------------------------------
    tempo_used = int(cfg.use_tempo_layer and bool(model.team_tempo_factor))
    tempo = 1.0
    if tempo_used:
        tempo = float(model.team_tempo_factor.get(home, 1.0)) * float(model.team_tempo_factor.get(away, 1.0))
        mu_h *= tempo
        mu_a *= tempo

    mu_total_after_tempo = mu_h + mu_a

    # ------------------------------------------------------------------
    # Soft league anchor (risk control) + safety clip
    # ------------------------------------------------------------------
    mu_total_pre_anchor = mu_h + mu_a
    mu_total = mu_total_pre_anchor

    w = float(cfg.soft_anchor_w_base)
    if mu_league > 0:
        if mu_total > mu_league:
            w = float(cfg.soft_anchor_w_upward)
        mu_total = (1.0 - w) * mu_total + w * float(mu_league)

    mu_total_post_anchor = mu_total

    clip_low = float(cfg.mu_league_anchor_low * mu_league) if mu_league > 0 else 0.0
    clip_high = float(cfg.mu_league_anchor_high * mu_league) if mu_league > 0 else 0.0
    if mu_league > 0:
        mu_total = float(np.clip(mu_total, cfg.mu_league_anchor_low * mu_league, cfg.mu_league_anchor_high * mu_league))

    mu_total_post_clip = mu_total

    # Restore split ratio (stable)
    split_final = float(np.clip(mu_h / max(1e-9, (mu_h + mu_a)), 0.30, 0.70))
    mu_h = mu_total * split_final
    mu_a = mu_total * (1.0 - split_final)

    # ------------------------------------------------------------------
    # Dispersion
    # ------------------------------------------------------------------
    alpha_base = float(model.alpha)
    alpha_used = float(alpha_for_match(model.alpha, cfg.alpha_squeeze))
    a = alpha_used

    # Simulate NB2 totals via gamma-poisson mixture for speed
    # NB2: mu, alpha => r=1/alpha, scale=alpha*mu
    # sample lambda ~ Gamma(shape=r, scale=alpha*mu); then k ~ Poisson(lambda)
    r = 1.0 / max(1e-12, a)

    lam_h = rng.gamma(shape=r, scale=a * mu_h, size=n_sims)
    lam_a = rng.gamma(shape=r, scale=a * mu_a, size=n_sims)

    k_h = rng.poisson(lam_h)
    k_a = rng.poisson(lam_a)

    sims = (k_h + k_a).astype(int)

    if not return_debug:
        return sims

    dbg: Dict[str, float] = {
        # base μ
        "mu_h0": float(mu_h0),
        "mu_a0": float(mu_a0),
        "mu_total_raw": float(mu_total_raw),
        "split0": float(split0),

        # hybrid
        "hybrid_used": float(hybrid_used),
        "mu_total_after_hybrid": float(mu_total_after_hybrid),

        # emp
        "emp_used": float(emp_used),
        "emp_fac_home": float(emp_fac_home),
        "emp_fac_away": float(emp_fac_away),
        "mu_total_after_emp": float(mu_total_after_emp),

        # mismatch
        "mismatch_used": float(mismatch_used),
        "mismatch_gap": float(gap),
        "mismatch_infl": float(infl),
        "mu_total_after_mismatch": float(mu_total_after_mismatch),

        # chaos
        "chaos_factor": float(chaos_factor),
        "chaos_add": float(chaos_add),
        "mu_total_after_chaos": float(mu_total_after_chaos),

        # tempo
        "tempo_used": float(tempo_used),
        "tempo_factor": float(tempo),
        "mu_total_after_tempo": float(mu_total_after_tempo),

        # anchor / clip
        "soft_anchor_w": float(w),
        "mu_total_pre_anchor": float(mu_total_pre_anchor),
        "mu_total_post_anchor": float(mu_total_post_anchor),
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
        "mu_total_post_clip": float(mu_total_post_clip),

        # final split
        "split_final": float(split_final),
        "mu_h_final": float(mu_h),
        "mu_a_final": float(mu_a),

        # dispersion
        "alpha_base": float(alpha_base),
        "alpha_used": float(alpha_used),
        "alpha_squeeze": float(cfg.alpha_squeeze),
    }

    return sims, dbg



def compute_fixture_odds(
    model: FittedModel,
    cfg: LeagueConfig,
    home: str,
    away: str,
    lines: List[float],
    margin: float,
    mu_league: float,
    over_price_boost: float,
    n_sims: int = 200_000,
) -> pd.DataFrame:
    gw_seed = 0
    rng = np.random.default_rng(stable_seed_from_match(home, away, gw_seed))

    sims, dbg = simulate_total_corners(model, cfg, home, away, mu_league, rng, n_sims=n_sims, return_debug=True)

    rows = []
    for line in lines:
        p_over = float(np.mean(sims > float(line)))
        odds_over, odds_under, imp_over, imp_under = bookmaker_odds_two_way(p_over, margin, over_price_boost)
        rows.append({
            "home": home,
            "away": away,
            "line": float(line),
            "p_over": p_over,
            "p_under": 1.0 - p_over,
            "bookmaker_odds_over": float(odds_over),
            "bookmaker_odds_under": float(odds_under),
            "imp_over": float(imp_over),
            "imp_under": float(imp_under),
            **dbg,
        })

    df = pd.DataFrame(rows)
    df["p_over_absdiff_5050"] = (df["p_over"] - 0.5).abs()
    best = df.loc[df["p_over_absdiff_5050"].idxmin()]
    df["is_main_line"] = df["line"] == float(best["line"])
    df["main_line"] = float(best["line"])

    # Estimate mu_match from sims for logging/inspection
    df["mu_match"] = float(np.mean(sims))
    df["mu_league"] = float(mu_league)
    df["alpha_used"] = float(alpha_for_match(model.alpha, cfg.alpha_squeeze))
    return df


# =============================================================================
# EXPORT TABLES
# =============================================================================

L2_LAMBDA_TEAM_BASE = 0.06
L2_LAMBDA_HOMEADV   = 0.01
EXPOSURE_EPS        = 5.0

def export_model_tables(model: FittedModel, df_train: pd.DataFrame, cfg: LeagueConfig, out_dir: str, tag: str, prefix: str) -> Tuple[str, str]:
    # Teams table (interpretable diagnostics)
    rows = []
    for t in model.teams:
        rows.append({
            "team": t,
            "attack": float(model.attack.get(t, 0.0)),
            "defense": float(model.defense.get(t, 0.0)),
            "matches_w": float(model.matches_w.get(t, 0.0)),
            "tempo": float(model.team_tempo_factor.get(t, 1.0)),
            "emp_against_factor": float(model.emp_against_factor.get(t, 1.0)),
        })
    df_teams = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Additional diagnostics derived from played matches
    #   (4) Home/Away empirical splits (for/against)
    #   (5) Expected (predicted) splits from the fitted model
    #   (6) Empirical variance of total corners per team
    # ------------------------------------------------------------------
    played = df_train[is_completed(df_train)].copy()

    if len(played) > 0:
        gw_num = pd.to_numeric(played.get(COL_GAMEWEEK, np.nan), errors="coerce")
        hc = pd.to_numeric(played.get(COL_HOME_CORNERS, np.nan), errors="coerce")
        ac = pd.to_numeric(played.get(COL_AWAY_CORNERS, np.nan), errors="coerce")

        # ensure valid (non-negative) counts only
        ok = gw_num.notna() & hc.notna() & ac.notna() & (hc >= 0) & (ac >= 0)
        played = played.loc[ok].copy()
        played[COL_HOME_CORNERS] = hc.loc[ok].astype(float)
        played[COL_AWAY_CORNERS] = ac.loc[ok].astype(float)

    if len(played) > 0:
        h = played[COL_HOME_TEAM].astype(str)
        a = played[COL_AWAY_TEAM].astype(str)
        hc = played[COL_HOME_CORNERS].astype(float)
        ac = played[COL_AWAY_CORNERS].astype(float)

        # predicted means for the played fixtures
        mu_h_pred = np.exp(model.beta0 + model.home_adv + h.map(model.attack).to_numpy() + a.map(model.defense).to_numpy())
        mu_a_pred = np.exp(model.beta0 + a.map(model.attack).to_numpy() + h.map(model.defense).to_numpy())

        # Build long form: each match contributes two rows (one per team)
        long_home = pd.DataFrame({
            "team": h.to_numpy(),
            "side": "home",
            "for_emp": hc.to_numpy(),
            "against_emp": ac.to_numpy(),
            "mu_for_pred": mu_h_pred,
            "mu_against_pred": mu_a_pred,
            "total_emp": (hc + ac).to_numpy(),
        })
        long_away = pd.DataFrame({
            "team": a.to_numpy(),
            "side": "away",
            "for_emp": ac.to_numpy(),
            "against_emp": hc.to_numpy(),
            "mu_for_pred": mu_a_pred,
            "mu_against_pred": mu_h_pred,
            "total_emp": (hc + ac).to_numpy(),
        })
        long = pd.concat([long_home, long_away], ignore_index=True)

        # home/away empirical means
        piv_emp = (
            long.pivot_table(index="team", columns="side", values=["for_emp", "against_emp"], aggfunc="mean")
            .reset_index()
        )
        # flatten columns
        piv_emp.columns = ["team"] + [f"{v}_{s}" for v, s in piv_emp.columns.tolist()[1:]]

        # home/away predicted means
        piv_pred = (
            long.pivot_table(index="team", columns="side", values=["mu_for_pred", "mu_against_pred"], aggfunc="mean")
            .reset_index()
        )
        piv_pred.columns = ["team"] + [f"{v}_{s}" for v, s in piv_pred.columns.tolist()[1:]]

        # overall predicted means (weighted by occurrences in data)
        overall = long.groupby("team").agg(
            mu_for_avg=("mu_for_pred", "mean"),
            mu_against_avg=("mu_against_pred", "mean"),
            total_emp_mean=("total_emp", "mean"),
            total_emp_var=("total_emp", lambda x: float(np.var(x.to_numpy(), ddof=1)) if len(x) > 1 else 0.0),
            n_obs=("total_emp", "size"),
        ).reset_index()
        overall["mu_total_team_avg"] = overall["mu_for_avg"] + overall["mu_against_avg"]

        df_teams = df_teams.merge(piv_emp, on="team", how="left")
        df_teams = df_teams.merge(piv_pred, on="team", how="left")
        df_teams = df_teams.merge(overall, on="team", how="left")

    # Keep legacy helper column (approx. total intensity implied by attack+defense)
    df_teams["mu_total_vs_avg"] = np.exp(model.beta0 + df_teams["attack"] + df_teams["defense"]) * 2.0

    os.makedirs(out_dir, exist_ok=True)

    teams_path = os.path.join(out_dir, f"{prefix}_model_teams_table_{tag}.csv")
    df_teams.to_csv(teams_path, index=False)

    # Sanity table

    played = df_train[is_completed(df_train)].copy()
    if len(played) > 0:
        mu_emp = float((played[COL_HOME_CORNERS] + played[COL_AWAY_CORNERS]).mean())
    else:
        mu_emp = 0.0

    df_sanity = pd.DataFrame([
        {"metric": "beta0", "value": float(model.beta0)},
        {"metric": "home_adv", "value": float(model.home_adv)},
        {"metric": "alpha", "value": float(model.alpha)},
        {"metric": "alpha_squeeze_used", "value": float(cfg.alpha_squeeze)},
        {"metric": "mu_league", "value": float(model.mu_league)},
        {"metric": "emp_total_corners_mean", "value": float(mu_emp)},
        {"metric": "USE_EMP_BIAS_CORRECTION", "value": int(cfg.use_emp_bias_correction)},
        {"metric": "EMP_BIAS_K", "value": float(cfg.emp_bias_k)},
        {"metric": "EMP_BIAS_MAX_FACTOR", "value": float(cfg.emp_bias_max_factor)},
        {"metric": "USE_HYBRID_INTERACTION", "value": int(cfg.use_hybrid_interaction)},
        {"metric": "HYBRID_W", "value": float(cfg.hybrid_w)},
        {"metric": "MISMATCH_INFLATE_ENABLED", "value": int(cfg.mismatch_inflate_enabled)},
        {"metric": "SOFT_ANCHOR_W_BASE", "value": float(cfg.soft_anchor_w_base)},
        {"metric": "SOFT_ANCHOR_W_UPWARD", "value": float(cfg.soft_anchor_w_upward)},
        {"metric": "CHAOS_MAX", "value": float(cfg.chaos_max)},
    ])

    sanity_path = os.path.join(out_dir, f"{prefix}_model_sanity_table_{tag}.csv")
    df_sanity.to_csv(sanity_path, index=False)

    return teams_path, sanity_path


# =============================================================================
# MAIN
# =============================================================================



def main(
    league: str = "EPL",
    margin: float = 0.08,
    over_price_boost: float = 0.01,
    alpha_squeeze: Optional[float] = None,
    n_sims: int = 200_000,
    target_gw: Optional[int] = None,
):
    if league not in LEAGUES:
        raise ValueError(f"Unknown league '{league}'. Choose one of: {LEAGUES}")

    log = setup_logging(tag=f"{league}")

    cfg = get_league_config(league, log, alpha_squeeze_override=alpha_squeeze)

    log.info("[%s] Using alpha_squeeze = %.3f", league, cfg.alpha_squeeze)


    patterns = MATCH_PATTERNS[league]
    files = find_match_files(DATA_DIR, patterns)
    if not files:
        log.error("No data files found for %s. Patterns: %s. DATA_DIR=%s", league, patterns, DATA_DIR)
        raise FileNotFoundError(f"No data files for {league} in {DATA_DIR}")

    # Load all seasons for the league
    weights = SEASON_WEIGHTS_BY_LEAGUE.get(league, {})
    dfs = []
    active_path = None

    for f in files:
        df = load_matches(f, league)
        w = season_weight_from_filename(f, weights) if weights else 1.0
        df["season_weight"] = float(w)
        dfs.append(df)

        if ACTIVE_SEASON_HINT in os.path.basename(f):
            active_path = f

    df_all = pd.concat(dfs, ignore_index=True)
    log.info("[%s] Loaded %d rows from %d files", league, len(df_all), len(files))

    if active_path is None:
        # fallback: pick newest file as active
        active_path = max(files, key=os.path.getmtime)
        log.warning("[%s] ACTIVE season not found via hint '%s'. Using newest file: %s",
                    league, ACTIVE_SEASON_HINT, os.path.basename(active_path))

    df_active = load_matches(active_path, league)
    df_active["season_weight"] = season_weight_from_filename(active_path, weights) if weights else 1.0

    # Train on all completed matches (all seasons with weights)
    df_train = df_all.copy()

    log.info("[%s] Training on completed matches...", league)
    model = fit_nb_team_model(df_train, log)

    # Compute emp bias + tempo factors
    if cfg.use_emp_bias_correction:
        model.emp_against_factor = compute_emp_against_factor(model, df_train, cfg, log)
    if cfg.use_tempo_layer:
        model.team_tempo_factor = compute_team_tempo_factor(model, df_train, cfg, log)

    # League mean from active played matches
    played = df_active[is_completed(df_active)].copy()
    if len(played) == 0:
        mu_league = float((df_train[is_completed(df_train)][COL_HOME_CORNERS] + df_train[is_completed(df_train)][COL_AWAY_CORNERS]).mean())
        log.warning("[%s] No played matches in active season → using train mean μ=%.3f", league, mu_league)
    else:
        mu_league = float((played[COL_HOME_CORNERS] + played[COL_AWAY_CORNERS]).mean())
        log.info("[%s] μ league (active played mean) = %.3f", league, mu_league)

    model.mu_league = float(mu_league)

    # Only completed matches define where we really are in the season
    played = df_active[df_active[COL_STATUS].isin(COMPLETED_STATUSES)].copy()

    gw_played = pd.to_numeric(played[COL_GAMEWEEK], errors="coerce").dropna().astype(int)

    if len(gw_played) == 0:
        raise ValueError("No completed matches found in active season.")

    last_played_gw = int(gw_played.max())
    current_gw = last_played_gw + 1

    log.info("[%s] Last played GW = %d → Pricing NEXT GW = %d", league, last_played_gw, current_gw)


    # Export tables
    teams_path, sanity_path = export_model_tables(model, df_train, cfg, DOUT_DIR, tag=f"gw{current_gw}", prefix=league)
    log.info("[%s] [OK] teams table: %s", league, teams_path)
    log.info("[%s] [OK] sanity table: %s", league, sanity_path)

    # Filter fixtures for current_gw in active season
    df_next = df_active[pd.to_numeric(df_active[COL_GAMEWEEK], errors="coerce") == current_gw]

    all_rows = []
    for _, r in df_next.iterrows():
        h = normalize_team_name(r[COL_HOME_TEAM], league)
        a = normalize_team_name(r[COL_AWAY_TEAM], league)

        if h not in model.attack or a not in model.attack:
            log.warning("[%s] Missing team params: %s or %s", league, h, a)
            continue

        # Use default ladder lines (or customize around mu_league if you want)
        lines = DEFAULT_LINES

        df_match = compute_fixture_odds(
            model,
            cfg,
            h,
            a,
            lines,
            margin,
            model.mu_league,
            over_price_boost,
            n_sims=n_sims
        )
        df_match = df_match.assign(gameweek=current_gw, home_team=h, away_team=a, league=league)
        all_rows.append(df_match)

    if not all_rows:
        log.warning("[%s] No fixtures produced. Check Game Week filtering.", league)
        return

    out = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(DOUT_DIR, f"{league}_bookmaker_odds_corners_gw{current_gw}_margin{margin:.3f}.csv")

    out.to_csv(out_path, index=False)
    log.info("[%s] Saved → %s (%d rows)", league, out_path, len(out))



# =============================================================================
# RUN ALL LEAGUES (module-level)
# =============================================================================
def run_all_leagues(
    margin: float = 0.08,
    over_price_boost: float = 0.01,
    alpha_squeeze=None,
    n_sims: int = 200_000,
):
    for lg in LEAGUES:
        try:
            main(
                league=lg,
                margin=margin,
                over_price_boost=over_price_boost,
                n_sims=n_sims,
            )
        except FileNotFoundError:
            print(f"[INFO] Skipping {lg} (no data found)")
            continue

if __name__ == "__main__":
    run_all_leagues(margin=0.08, over_price_boost=0.01, alpha_squeeze=None, n_sims=50_000)


