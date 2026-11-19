"""GTA VI trophy rarity forecaster for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (0.05/0.25/0.5/0.75/
0.95) on the percent scale.

Model (standard library only, deterministic, no network, no randomness):

  * Work on the logit scale.  A trophy's earn rate is the share of the players
    Exophase tracks who have earned it, so it moves with two forces: the
    tracked population N(t) grows (dilution by newcomers who have earned
    little, and progression as existing players advance) and time passes.
  * Per trophy, fit two least-squares lines on the case's history, each
    anchored at the last read so the intercept is the recent level:
      A) logit(rate) ~ a + b * ln(N / N_last)   (population / dilution trend)
      B) logit(rate) ~ a + c * (ln(1 + hours) - ln(1 + hours_last))
    Slopes are shrunk toward the pooled (median) slope across trophies, so a
    trophy with a flat or noisy history borrows the common trend instead of
    extrapolating noise.
  * N at the target is forecast by a power law in time since release,
    ln N ~ p + q * ln(1 + hours); an exponential fit supplies only the
    uncertainty in ln N_target (on accelerating data it explodes, so it must
    never drag the median).
  * The median is the average of the two models' predictions where both are
    fittable.  Where neither is (e.g. the practice cases, whose whole history
    is one unchanged launch-night page), a prior applies: rates fall on a
    refresh as newcomers dilute them, so the median sits a little below the
    last rate, with a wide spread.
  * The spread combines, in quadrature: the baseline random-walk term
    (0.5 * sqrt(days)), the fit residuals, slope uncertainty times the
    extrapolation distance, the slope times the uncertainty in ln N_target,
    and a refresh-jump term (Exophase refreshes in discrete jumps; the answer
    is the page as of its last refresh before the target).
  * The 100% trophy (every tracked player has it) is special: its median is
    99.7% with a modest spread, since newcomers earn it within minutes of
    starting, while a small share of tracked players at any refresh have not
    reached it yet.

Output: /output/<id>.out, one object {"trophies": [{"id": n, "q": [...]}]},
one row per trophy, quantiles non-decreasing, each in [0.01, 99.99], rounded
to 4 decimals half-up.  Reads every <id>.in in $MAKEFUN_INPUT_DIR (else
/input), writes to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# Random-walk spread of a rate's move, in logit units per square-root day
# (the house baseline's sigma).
SIGMA_WALK = 0.5
# Spread added for the discrete Exophase refresh jumps.
SIGMA_REFRESH = 0.30
# Prior median logit shift when history is flat (practice cases): rates fall
# on a refresh as newcomers dilute them.
PRIOR_SHIFT = -0.30
# Prior spread when history is flat: the refresh move is largely unknown.
PRIOR_SPREAD = 1.0
# Shrinkage strength for per-trophy slopes toward the pooled slope.
SHRINK_K = 4.0
# History thinning: at most one read per this many minutes, for fitting.
THIN_MINUTES = 30.0
# Cap on the extrapolated tracked-players multiple over the last read.
N_GROWTH_CAP = 100.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    """Parse an ISO 8601 timestamp (with Z or offset) to an aware datetime."""
    if isinstance(s, (int, float)):
        return datetime.fromtimestamp(s, tz=timezone.utc)
    s = str(s).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_release(game):
    """Release instant; the game released 2026-11-19 (UTC)."""
    raw = (game or {}).get("release_date") or (game or {}).get("release")
    if raw:
        try:
            s = str(raw).strip()
            if len(s) == 10:
                return datetime.fromisoformat(s + "T00:00:00+00:00")
            return parse_time(s)
        except ValueError:
            pass
    return datetime(2026, 11, 19, tzinfo=timezone.utc)


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std, slope_se) or None."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    var = sum(r * r for r in resid) / dof
    resid_std = math.sqrt(var)
    slope_se = math.sqrt(var / sxx) if sxx > 0 else 0.0
    return (a, b, resid_std, slope_se)


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def thin_reads(reads, release):
    """One read per THIN_MINUTES bucket (the last in each bucket)."""
    kept = []
    last_bucket = None
    for r in reads:
        try:
            t = parse_time(r["at"])
        except (KeyError, ValueError, TypeError):
            continue
        bucket = int((t - release).total_seconds() // (THIN_MINUTES * 60))
        if bucket != last_bucket:
            kept.append(r)
            last_bucket = bucket
        else:
            kept[-1] = r
    return kept


def forecast_case(case, name):
    trophies = case["trophies"]
    history = case.get("history") or []
    if not trophies:
        raise ValueError(f"{name} lists no trophies")
    if not history:
        raise ValueError(f"{name} has no history")
    release = parse_release(case.get("game") or {})
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    hours_target = max((target - release).total_seconds() / 3600.0, 0.0)
    u_target = math.log1p(hours_target)

    thin = thin_reads(history, release)
    if not thin:
        thin = history
    last = thin[-1]
    n_last = float(last["players_tracked"])
    t_last = parse_time(last["at"])
    hours_last = max((t_last - release).total_seconds() / 3600.0, 0.0)
    u_last = math.log1p(hours_last)

    # --- Forecast N at the target (two growth laws, geometric mean). ---
    ln_n_target = math.log(n_last)
    sigma_ln_n = 0.35
    xs_u, xs_h, ys_n = [], [], []
    for r in thin:
        try:
            t = parse_time(r["at"])
            n = float(r["players_tracked"])
        except (KeyError, ValueError, TypeError):
            continue
        if n <= 0:
            continue
        h = max((t - release).total_seconds() / 3600.0, 0.0)
        xs_u.append(math.log1p(h))
        xs_h.append(h)
        ys_n.append(math.log(n))
    fit_u = ols(xs_u, ys_n)
    fit_h = ols(xs_h, ys_n)
    pred_u = fit_u[0] + fit_u[1] * u_target if fit_u is not None else None
    pred_h = fit_h[0] + fit_h[1] * hours_target if fit_h is not None else None
    if pred_u is not None:
        # Central: the power law in time since release. Launch-week tracking
        # growth decelerates; the exponential fit only widens the uncertainty,
        # it must not drag the median up (on accelerating data it explodes).
        ln_n_target = pred_u
        sigma_ln_n = 0.25
        if pred_h is not None:
            sigma_ln_n = 0.5 * abs(pred_h - pred_u) + 0.25
    elif pred_h is not None:
        ln_n_target = pred_h
        sigma_ln_n = 0.45
    if True:
        ln_n_target = min(ln_n_target, math.log(n_last) + math.log(N_GROWTH_CAP))
        ln_n_target = max(ln_n_target, math.log(n_last))
    dx_n = ln_n_target - math.log(n_last)
    du = u_target - u_last

    # --- Per-trophy series on the logit scale. ---
    series = []
    for i in range(len(trophies)):
        pts = []
        for r in thin:
            rates = r.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            try:
                t = parse_time(r["at"])
                n = float(r["players_tracked"])
                rate = float(rates[i])
            except (KeyError, ValueError, TypeError):
                continue
            if n <= 0:
                continue
            h = max((t - release).total_seconds() / 3600.0, 0.0)
            pts.append((math.log(n), math.log1p(h), logit(rate)))
        series.append(pts)

    # Last non-null rate per trophy (fallback for the level).
    last_rates = []
    for i in range(len(trophies)):
        rate = None
        for r in reversed(history):
            rates = r.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                break
        last_rates.append(rate if rate is not None else 0.0)

    # --- Pooled slopes across trophies (shrinkage targets). ---
    slopes_n, slopes_u = [], []
    fits_n, fits_u = [], []
    for pts in series:
        fn = ols([p[0] - math.log(n_last) for p in pts], [p[2] for p in pts])
        fu = ols([p[1] - u_last for p in pts], [p[2] for p in pts])
        fits_n.append(fn)
        fits_u.append(fu)
        if fn is not None:
            slopes_n.append(fn[1])
        if fu is not None:
            slopes_u.append(fu[1])
    pooled_n = median(slopes_n) if slopes_n else 0.0
    pooled_u = median(slopes_u) if slopes_u else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        lg_last = logit(rate)
        fn = fits_n[i]
        fu = fits_u[i]
        n_pts = len(series[i])

        # Shrink each slope toward the pooled slope.
        def shrink(fit, pooled):
            if fit is None:
                return None
            w = n_pts / (n_pts + SHRINK_K)
            return (fit[0], w * fit[1] + (1.0 - w) * pooled,
                    fit[2], fit[3])

        fn_s = shrink(fn, pooled_n)
        fu_s = shrink(fu, pooled_u)

        # A page that literally never changed (no rate move, no N move) teaches
        # nothing about movement: the time-trend fit would report slope 0 with
        # zero residual and claim false confidence. Such a trophy takes the
        # prior path instead (median shifted down, wide spread).
        pts = series[i]
        page_frozen = False
        if pts:
            ys = [p[2] for p in pts]
            ns = [p[0] for p in pts]
            page_frozen = (max(ys) - min(ys) < 1e-9
                           and max(ns) - min(ns) < 1e-9)
        if page_frozen:
            fn_s = None
            fu_s = None

        medians = []
        spread2 = SIGMA_REFRESH ** 2 + (SIGMA_WALK ** 2) * horizon_days
        if fn_s is not None:
            medians.append(fn_s[0] + fn_s[1] * dx_n)
            spread2 += max(fn_s[2], 0.15) ** 2
            spread2 += (fn_s[3] * abs(dx_n)) ** 2
            spread2 += (fn_s[1] * sigma_ln_n) ** 2
        if fu_s is not None:
            medians.append(fu_s[0] + fu_s[1] * du)
            spread2 += max(fu_s[2], 0.15) ** 2
            spread2 += (fu_s[3] * abs(du)) ** 2

        if rate >= 100.0:
            # Everybody has it; newcomers earn it within minutes of starting.
            med = logit(99.7)
            spread = 0.45
        elif medians:
            med = sum(medians) / len(medians)
            spread = math.sqrt(max(spread2, 0.0))
        else:
            # Flat history (practice): a refresh is coming that dilutes rates.
            med = lg_last + PRIOR_SHIFT * (1.0 - rate / 100.0)
            spread = math.sqrt(PRIOR_SPREAD ** 2 + (SIGMA_WALK ** 2) * horizon_days)

        spread = min(max(spread, 0.35), 2.2)
        qs = []
        prev = None
        for z in Z:
            q = clamp_pct(sigmoid(med + z * spread))
            q = round4(q)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_forecast(case):
    """A valid-shaped forecast used only if the model path fails."""
    rows = []
    last = (case.get("history") or [{}])[-1]
    rates = last.get("rates") or []
    for i, trophy in enumerate(case.get("trophies") or []):
        rate = rates[i] if i < len(rates) and rates[i] is not None else 1.0
        med = logit(rate if rate < 100.0 else 99.7)
        qs = []
        prev = None
        for z in Z:
            q = round4(clamp_pct(sigmoid(med + z * 1.0)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = fallback_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
