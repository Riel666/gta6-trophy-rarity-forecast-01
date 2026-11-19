# gta6-trophy-rarity-forecast-01 — Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Riel666/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Riel666/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for make.fun's **gta6-trophy-rarity-forecast-01** bounty ("Forecast how
many GTA VI players earn each trophy in week one", $250, top 20 share).

## What it does

`main.py` is a dependency-free Python 3 program (standard library only, no
network, no randomness). Given one case file (`/input/<id>.in`, the house's
recording of Exophase's GTA VI PS5 trophy page up to the case's checkpoint),
it forecasts every trophy's earn rate at the case's target and writes
`/output/<id>.out`: one row per trophy with five quantiles (0.05 / 0.25 /
0.5 / 0.75 / 0.95) of the rate in percent.

The model works on the logit scale. Per trophy it fits two least-squares
lines on the case's history, anchored at the last read: one against
log tracked players (the dilution trend as newcomers join) and one against
log time since release, with slopes shrunk toward the pooled slope across
trophies. Tracked players at the target come from a power law in time since
release; an exponential fit supplies only the uncertainty. The spread
combines the random-walk term, fit residuals, slope uncertainty, the
uncertainty in the player count, and a term for Exophase's discrete refresh
jumps. Where history is a single unchanged page (the practice cases), a
prior applies: a refresh is coming that dilutes rates, so the median sits
below the last rate with a wide spread.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/<id>.out for every <id>.in found
```

## Validation

Local runs on the kit's practice cases (`p00`, `p01`) produce valid
forecasts: 55 rows each, five non-decreasing quantiles per row, every value
in [0.01, 99.99]. On a synthetic final-like case (history extended with
tracked players growing and rates following a log-player trend) the program
scores a mean pinball loss of 0.294 against 0.346 for the house baseline
starter.

---

<sub>Entered on [make.fun](https://make.fun/?ref=nrw7bbqr&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
