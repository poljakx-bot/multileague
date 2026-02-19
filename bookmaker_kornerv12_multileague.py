import os
import glob
import math
import logging
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

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
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
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
        SOFT_ANCHOR_W_BASE=0.06,
        SOFT_ANCHOR_W_UPWARD=0.14,
        CHAOS_MAX=0.03,
    ),
    "SA": dict(
        EMP_BIAS_K=30.0,
        EMP_BIAS_MAX_FACTOR=1.25,
        SOFT_ANCHOR_W_BASE=0.06,
        SOFT_ANCHOR_W_UPWARD=0.14,
        CHAOS_MAX=0.025,
    ),
    "BL": dict(
        EMP_BIAS_K=28.0,
        EMP_BIAS_MAX_FACTOR=1.28,
        SOFT_ANCHOR_W_BASE=0.07,
        SOFT_ANCHOR_W_UPWARD=0.16,
        CHAOS_MAX=0.035,
    ),
    "L1": dict(
        EMP_BIAS_K=30.0,
        EMP_BIAS_MAX_FACTOR=1.25,
        SOFT_ANCHOR_W_BASE=0.06,
        SOFT_ANCHOR_W_UPWARD=0.14,
        CHAOS_MAX=0.03,
    ),
}

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

TEAM_ALIASES = {
    # common normalizations (extend as needed)
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham",
    "Nottingham Forest": "Nott'm Forest",
    "Newcastle United": "Newcastle",
    # La Liga accents
    "Atlético Madrid": "Atletico Madrid",
    "Deportivo Alavés": "Alaves",
    "Celta de Vigo": "Celta Vigo",
    "Real Sociedad de Fútbol": "Real Sociedad",
}

def normalize_team_name(x: str) -> str:
    s = str(x).strip()
    s = TEAM_ALIASES.get(s, s)
    return s


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


def load_matches(path: str) -> pd.DataFrame:
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

    # Normalize team names
    df[COL_HOME_TEAM] = df[COL_HOME_TEAM].astype(str).map(normalize_team_name)
    df[COL_AWAY_TEAM] = df[COL_AWAY_TEAM].astype(str).map(normalize_team_name)

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


def nb2_loglik(k: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    # NB2 parameterization: Var = mu + alpha*mu^2
    alpha = float(np.clip(alpha, 1e-12, 1e3))
    r = 1.0 / alpha
    p = r / (r + mu)
    # log PMF
    return float(np.sum(gammaln(k + r) - gammaln(r) - gammaln(k + 1) + r * np.log(p) + k * np.log(1 - p)))


def fit_nb_team_model(df: pd.DataFrame, log: logging.Logger) -> FittedModel:
    # Build team list
    teams = sorted(list(set(df[COL_HOME_TEAM]).union(set(df[COL_AWAY_TEAM]))))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # Observations (only completed)
    d = df[is_completed(df)].copy()
    d = d.dropna(subset=[COL_HOME_CORNERS, COL_AWAY_CORNERS, COL_GAMEWEEK])
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

        mu_h = np.exp(beta0 + home_adv + a[home_i] + dfn[away_i])
        mu_a = np.exp(beta0 + a[away_i] + dfn[home_i])

        # Weighted loglik
        ll = 0.0
        ll += np.sum(w * (np.vectorize(lambda k, mu: nb2_loglik(np.array([k]), np.array([mu]), alpha))(y_home, mu_h)))
        ll += np.sum(w * (np.vectorize(lambda k, mu: nb2_loglik(np.array([k]), np.array([mu]), alpha))(y_away, mu_a)))

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

def compute_emp_against_factor(model: FittedModel, df_train: pd.DataFrame, log: logging.Logger) -> Dict[str, float]:
    # Empirical conceded corners vs model predicted conceded corners
    d = df_train[is_completed(df_train)].copy()
    d = d.dropna(subset=[COL_HOME_CORNERS, COL_AWAY_CORNERS])
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

        # shrink toward 1.0 using EMP_BIAS_K
        shrink = counts[t] / (counts[t] + float(EMP_BIAS_K))
        f = 1.0 + shrink * (raw - 1.0)

        # clamp
        f = float(np.clip(f, 1.0 / EMP_BIAS_MAX_FACTOR, EMP_BIAS_MAX_FACTOR))
        fac[t] = f

    log.info("Emp-bias: mean factor=%.3f, min=%.3f, max=%.3f",
             float(np.mean(list(fac.values()))),
             float(np.min(list(fac.values()))),
             float(np.max(list(fac.values()))))
    return fac


def compute_team_tempo_factor(model: FittedModel, df_train: pd.DataFrame, log: logging.Logger) -> Dict[str, float]:
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

        shrink = cnt_team[t] / (cnt_team[t] + float(TEMPO_K))
        f = 1.0 + shrink * (raw - 1.0)
        f = float(np.clip(f, 1.0 / TEMPO_MAX_FACTOR, TEMPO_MAX_FACTOR))
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
    home: str,
    away: str,
    mu_league: float,
    alpha_squeeze: float,
    rng: np.random.Generator,
    n_sims: int = 50_000,
) -> np.ndarray:
    # Base μ
    mu_h = math.exp(model.beta0 + model.home_adv + model.attack.get(home, 0.0) + model.defense.get(away, 0.0))
    mu_a = math.exp(model.beta0 + model.attack.get(away, 0.0) + model.defense.get(home, 0.0))

    # Hybrid interaction (optional)
    if USE_HYBRID_INTERACTION:
        mu_total = mu_h + mu_a
        # mild additive blending toward league mean shape
        mu_total_hybrid = HYBRID_W * mu_total + (1.0 - HYBRID_W) * (0.5 * mu_total + 0.5 * mu_league)
        # keep split ratio stable
        split = float(np.clip(mu_h / max(1e-9, mu_total), 0.30, 0.70))
        mu_h = mu_total_hybrid * split
        mu_a = mu_total_hybrid * (1.0 - split)

    # Emp-bias correction (optional)
    if USE_EMP_BIAS_CORRECTION and model.emp_against_factor:
        mu_h *= float(model.emp_against_factor.get(away, 1.0))
        mu_a *= float(model.emp_against_factor.get(home, 1.0))

    # Mismatch inflation (optional)
    if MISMATCH_INFLATE_ENABLED:
        gap = abs(float(model.attack.get(home, 0.0)) - float(model.attack.get(away, 0.0)))
        if gap > MISMATCH_GAP_THRESHOLD:
            infl = 1.0 + min(MISMATCH_INFLATE_CAP, MISMATCH_INFLATE_SLOPE * (gap - MISMATCH_GAP_THRESHOLD))
            mu_h *= infl
            mu_a *= infl

    # ------------------------------------------------------------------
    # CHAOS RELEASE (league-controlled)
    # Corrects NB bias where low-possession teams look like low-event.
    # ------------------------------------------------------------------
    away_att = float(model.attack.get(away, 0.0))
    away_def = float(model.defense.get(away, 0.0))
    home_att = float(model.attack.get(home, 0.0))
    home_def = float(model.defense.get(home, 0.0))

    chaos_factor = 1.00

    # Soft chaos scaling (0..CHAOS_MAX) depending on how close defense is to 0
    if away_att < CHAOS_ATTACK_THRESH and abs(away_def) < CHAOS_DEF_NEAR0_BAND:
        def_near0 = 1.0 - (abs(away_def) / CHAOS_DEF_NEAR0_BAND)  # 1 at def=0, 0 at band edge
        chaos_factor = 1.00 + float(CHAOS_MAX) * float(def_near0)
    elif home_att < CHAOS_ATTACK_THRESH and abs(home_def) < CHAOS_DEF_NEAR0_BAND:
        def_near0 = 1.0 - (abs(home_def) / CHAOS_DEF_NEAR0_BAND)
        chaos_factor = 1.00 + float(CHAOS_MAX * 0.8) * float(def_near0)

    mu_h *= chaos_factor
    mu_a *= chaos_factor

    # Tempo layer (optional)
    if USE_TEMPO_LAYER and model.team_tempo_factor:
        tempo = float(model.team_tempo_factor.get(home, 1.0)) * float(model.team_tempo_factor.get(away, 1.0))
        mu_h *= tempo
        mu_a *= tempo

    # Soft league anchor (risk control)
    mu_total = mu_h + mu_a
    if mu_league > 0:
        w = float(SOFT_ANCHOR_W_BASE)
        if mu_total > mu_league:
            w = float(SOFT_ANCHOR_W_UPWARD)
        mu_total = (1.0 - w) * mu_total + w * float(mu_league)

    # Safety clip around league band
    if mu_league > 0:
        mu_total = float(np.clip(mu_total, MU_LEAGUE_ANCHOR_LOW * mu_league, MU_LEAGUE_ANCHOR_HIGH * mu_league))

    # Restore split ratio (stable)
    split = float(np.clip(mu_h / max(1e-9, (mu_h + mu_a)), 0.30, 0.70))
    mu_h = mu_total * split
    mu_a = mu_total * (1.0 - split)

    # Dispersion
    a = alpha_for_match(model.alpha, alpha_squeeze)

    # Simulate NB2 totals via gamma-poisson mixture for speed
    # NB2: mu, alpha => r=1/alpha, scale=alpha*mu
    # sample lambda ~ Gamma(shape=r, scale=alpha*mu); then k ~ Poisson(lambda)
    r = 1.0 / max(1e-12, a)

    lam_h = rng.gamma(shape=r, scale=a * mu_h, size=n_sims)
    lam_a = rng.gamma(shape=r, scale=a * mu_a, size=n_sims)

    k_h = rng.poisson(lam_h)
    k_a = rng.poisson(lam_a)

    return (k_h + k_a).astype(int)


def compute_fixture_odds(
    model: FittedModel,
    home: str,
    away: str,
    lines: List[float],
    margin: float,
    mu_league: float,
    over_price_boost: float,
    alpha_squeeze: float,
    n_sims: int = 200_000,
) -> pd.DataFrame:
    gw_seed = 0
    rng = np.random.default_rng(stable_seed_from_match(home, away, gw_seed))

    sims = simulate_total_corners(model, home, away, mu_league, alpha_squeeze, rng, n_sims=n_sims)

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
        })

    df = pd.DataFrame(rows)
    df["p_over_absdiff_5050"] = (df["p_over"] - 0.5).abs()
    best = df.loc[df["p_over_absdiff_5050"].idxmin()]
    df["is_main_line"] = df["line"] == float(best["line"])
    df["main_line"] = float(best["line"])

    # Estimate mu_match from sims for logging/inspection
    df["mu_match"] = float(np.mean(sims))
    df["mu_league"] = float(mu_league)
    df["alpha_used"] = float(alpha_for_match(model.alpha, alpha_squeeze))
    return df


# =============================================================================
# EXPORT TABLES
# =============================================================================

L2_LAMBDA_TEAM_BASE = 0.06
L2_LAMBDA_HOMEADV   = 0.01
EXPOSURE_EPS        = 5.0

def export_model_tables(model: FittedModel, df_train: pd.DataFrame, out_dir: str, tag: str, prefix: str) -> Tuple[str, str]:
    # Teams table
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
    df_teams["mu_total_vs_avg"] = np.exp(model.beta0 + df_teams["attack"] + df_teams["defense"]) * 2.0

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
        {"metric": "mu_league", "value": float(model.mu_league)},
        {"metric": "emp_total_corners_mean", "value": float(mu_emp)},
        {"metric": "USE_EMP_BIAS_CORRECTION", "value": int(USE_EMP_BIAS_CORRECTION)},
        {"metric": "EMP_BIAS_K", "value": float(EMP_BIAS_K)},
        {"metric": "EMP_BIAS_MAX_FACTOR", "value": float(EMP_BIAS_MAX_FACTOR)},
        {"metric": "USE_HYBRID_INTERACTION", "value": int(USE_HYBRID_INTERACTION)},
        {"metric": "HYBRID_W", "value": float(HYBRID_W)},
        {"metric": "MISMATCH_INFLATE_ENABLED", "value": int(MISMATCH_INFLATE_ENABLED)},
        {"metric": "SOFT_ANCHOR_W_BASE", "value": float(SOFT_ANCHOR_W_BASE)},
        {"metric": "SOFT_ANCHOR_W_UPWARD", "value": float(SOFT_ANCHOR_W_UPWARD)},
        {"metric": "CHAOS_MAX", "value": float(CHAOS_MAX)},
    ])

    sanity_path = os.path.join(out_dir, f"{prefix}_model_sanity_table_{tag}.csv")
    df_sanity.to_csv(sanity_path, index=False)

    return teams_path, sanity_path


# =============================================================================
# MAIN
# =============================================================================

def apply_league_preset(league: str, log: logging.Logger) -> None:
    global EMP_BIAS_K, EMP_BIAS_MAX_FACTOR, SOFT_ANCHOR_W_BASE, SOFT_ANCHOR_W_UPWARD, CHAOS_MAX

    if league not in LEAGUE_PRESETS:
        log.warning("No league preset for %s, using current globals.", league)
        return

    p = LEAGUE_PRESETS[league]
    EMP_BIAS_K = float(p["EMP_BIAS_K"])
    EMP_BIAS_MAX_FACTOR = float(p["EMP_BIAS_MAX_FACTOR"])
    SOFT_ANCHOR_W_BASE = float(p["SOFT_ANCHOR_W_BASE"])
    SOFT_ANCHOR_W_UPWARD = float(p["SOFT_ANCHOR_W_UPWARD"])
    CHAOS_MAX = float(p["CHAOS_MAX"])

    log.info("Applied preset %s: EMP_BIAS_K=%.1f, EMP_BIAS_MAX_FACTOR=%.2f, SOFT_ANCHOR_W_BASE=%.3f, SOFT_ANCHOR_W_UPWARD=%.3f, CHAOS_MAX=%.3f",
             league, EMP_BIAS_K, EMP_BIAS_MAX_FACTOR, SOFT_ANCHOR_W_BASE, SOFT_ANCHOR_W_UPWARD, CHAOS_MAX)


def main(
    league: str = "EPL",
    margin: float = 0.08,
    over_price_boost: float = 0.01,
    alpha_squeeze: float = ALPHA_SQUEEZE_DEFAULT,
    n_sims: int = 200_000,
):
    if league not in LEAGUES:
        raise ValueError(f"Unknown league '{league}'. Choose one of: {LEAGUES}")

    log = setup_logging(tag=f"{league}")

    apply_league_preset(league, log)

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
        df = load_matches(f)
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

    df_active = load_matches(active_path)
    df_active["season_weight"] = season_weight_from_filename(active_path, weights) if weights else 1.0

    # Train on all completed matches (all seasons with weights)
    df_train = df_all.copy()

    log.info("[%s] Training on completed matches...", league)
    model = fit_nb_team_model(df_train, log)

    # Compute emp bias + tempo factors
    if USE_EMP_BIAS_CORRECTION:
        model.emp_against_factor = compute_emp_against_factor(model, df_train, log)
    if USE_TEMPO_LAYER:
        model.team_tempo_factor = compute_team_tempo_factor(model, df_train, log)

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
    teams_path, sanity_path = export_model_tables(model, df_train, DATA_DIR, tag=f"gw{current_gw}", prefix=league)
    log.info("[%s] [OK] teams table: %s", league, teams_path)
    log.info("[%s] [OK] sanity table: %s", league, sanity_path)

    # Filter fixtures for current_gw in active season
    df_next = df_active[pd.to_numeric(df_active[COL_GAMEWEEK], errors="coerce") == current_gw]

    all_rows = []
    for _, r in df_next.iterrows():
        h = normalize_team_name(r[COL_HOME_TEAM])
        a = normalize_team_name(r[COL_AWAY_TEAM])

        if h not in model.attack or a not in model.attack:
            log.warning("[%s] Missing team params: %s or %s", league, h, a)
            continue

        # Use default ladder lines (or customize around mu_league if you want)
        lines = DEFAULT_LINES

        df_match = compute_fixture_odds(
            model, h, a, lines, margin, mu_league, over_price_boost, alpha_squeeze, n_sims=n_sims
        )
        df_match = df_match.assign(gameweek=current_gw, home_team=h, away_team=a, league=league)
        all_rows.append(df_match)

    if not all_rows:
        log.warning("[%s] No fixtures produced. Check Game Week filtering.", league)
        return

    out = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(DATA_DIR, f"{league}_bookmaker_odds_corners_gw{current_gw}_margin{margin:.3f}.csv")
    out.to_csv(out_path, index=False)
    log.info("[%s] Saved → %s (%d rows)", league, out_path, len(out))


def run_all_leagues(
    margin: float = 0.08,
    over_price_boost: float = 0.01,
    alpha_squeeze: float = ALPHA_SQUEEZE_DEFAULT,
    n_sims: int = 200_000,
):
    # Convenience runner: loops across all leagues that have data present
    for lg in LEAGUES:
        try:
            main(lg, margin=margin, over_price_boost=over_price_boost, alpha_squeeze=alpha_squeeze, n_sims=n_sims)
        except FileNotFoundError:
            # skip leagues without files
            continue


if __name__ == "__main__":
    run_all_leagues(margin=0.08, over_price_boost=0.01, alpha_squeeze=ALPHA_SQUEEZE_DEFAULT, n_sims=50_000)

