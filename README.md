# broadband-wholesale-establishments

Replication code for a county-level study of broadband adoption and wholesale establishment structure, 2008–2023.

## Files

- `build_controls.py` — builds `controls.csv` (county population from Census PEP, metro status from USDA ERS Rural-Urban Continuum Codes 2013)
- `run_tables.py` — produces Tables 2–6 from `wholesale_panel.csv` and `controls.csv`
- `ols_assumptions.py` — regression diagnostics (VIF, Breusch–Pagan, Jarque–Bera, state-clustered errors, Cook's distance)
- `wholesale_panel.csv` — county × industry-group establishment and employment counts from County Business Patterns, with FCC Form 477 broadband tiers. Assembled separately; that build code is not in this repo.

## Running

    python3 run_tables.py

`controls.csv` is included, so this runs as-is. To rebuild it, download the 2013 Rural-Urban Continuum Codes from https://www.ers.usda.gov/data-products/rural-urban-continuum-codes into this folder and run `build_controls.py` first.

## Specification

    d_lestab = b0 + b1*d_bb + b2*bb_base + b3*lestab_base + b4*d_lpop + b5*metro + e

Long-difference regression: each county contributes one observation of the 2008–2023 change.

## Notes

`MIN_ESTAB = 5` excludes counties with fewer than five wholesale establishments in the start year. The log outcome makes tiny counties swing hard — 4 falling to 2 is the same −0.69 as 400 falling to 200.

RUCC 2013 predates Connecticut's 2022 switch to planning regions, so Connecticut counties drop from the controls merge.
