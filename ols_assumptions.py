import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor, OLSInfluence)
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, linear_reset)
from statsmodels.stats.stattools import jarque_bera

HERE = os.path.dirname(os.path.abspath(__file__))

BB = "bb10"          # switch to "bb10"
OUTCOME = "d_lestab"  # switch to "d_lsize"
FULL = ["d_bb", "bb_base", "lestab_base", "d_lpop", "metro"]
MIN_ESTAB = 5


def find(name):
    for path in [os.path.join(HERE, name),
                 os.path.join(HERE, "Data", name),
                 os.path.join(os.path.dirname(HERE), name),
                 os.path.join(os.path.dirname(HERE), "Data", name)]:
        if os.path.isfile(path):
            return path
    print(f"\nCould not find {name}. Run 01_build_controls.py first.")
    sys.exit(1)


def build():
    panel = pd.read_csv(find("wholesale_panel.csv"), dtype={"fips": str})
    panel["fips"] = panel["fips"].str.strip().str.zfill(5)

    controls = pd.read_csv(find("controls.csv"), dtype={"fips": str})
    controls["fips"] = controls["fips"].str.strip().str.zfill(5)

    groups = list(panel["group"].unique())
    aggregate = next((g for g in groups
                      if str(g).lower() in ("all", "total", "wholesale", "42")),
                     None)
    if aggregate is None:
        print(f"No aggregate group found. Groups: {groups}")
        sys.exit(1)

    df = panel[panel["group"] == aggregate].copy()
    df = df.merge(controls, on="fips", how="left")

    for col in ["estab_2008", "estab_2023", "pop2008", "pop2023"]:
        df.loc[df[col] <= 0, col] = np.nan

    if MIN_ESTAB > 0:
        df = df[df["estab_2008"] >= MIN_ESTAB].copy()

    df["d_lestab"] = np.log(df["estab_2023"]) - np.log(df["estab_2008"])
    df["lestab_base"] = np.log(df["estab_2008"])
    df["d_lpop"] = np.log(df["pop2023"]) - np.log(df["pop2008"])
    df["d_bb"] = df[f"{BB}_2023"] - df[f"{BB}_2008"]
    df["bb_base"] = df[f"{BB}_2008"]

    df["size_08"] = df["emp_2008"] / df["estab_2008"]
    df["size_23"] = df["emp_2023"] / df["estab_2023"]
    for col in ["size_08", "size_23"]:
        df.loc[df[col] <= 0, col] = np.nan
    df["d_lsize"] = np.log(df["size_23"]) - np.log(df["size_08"])

    return df.dropna(subset=[OUTCOME] + FULL)


def section(n, title):
    print(f"\n{'-' * 70}\n{n}. {title}\n{'-' * 70}")


def main():
    df = build()
    X = sm.add_constant(df[FULL], has_constant="add")
    y = df[OUTCOME]

    ols = sm.OLS(y, X).fit()                 # classical, for tests
    hc3 = sm.OLS(y, X).fit(cov_type="HC3")   # as reported in the paper

    print(f"Specification: {OUTCOME} on {', '.join(FULL)}  [{BB}]")
    print(f"n = {int(ols.nobs)}, R-squared = {ols.rsquared:.4f}")

    # ---- 1. Multicollinearity ----
    section(1, "MULTICOLLINEARITY — variance inflation factors")
    for i, name in enumerate(X.columns):
        if name == "const":
            continue
        print(f"  {name:16s} VIF = {variance_inflation_factor(X.values, i):6.2f}")
    print("\n  Above 10 signals a problem; above 5 is worth noting. Correlated")
    print("  regressors inflate standard errors but leave coefficients unbiased.")

    # ---- 2. Functional form ----
    section(2, "FUNCTIONAL FORM — Ramsey RESET")
    reset = linear_reset(ols, power=2, use_f=True)
    print(f"  F = {reset.fvalue:.3f}, p = {reset.pvalue:.4f}")
    print("\n  A low p-value suggests omitted nonlinearity: squared or")
    print("  interaction terms would improve fit. Since the broadband measure")
    print("  is ordinal, a rejection here may reflect unequal spacing between")
    print("  tiers rather than a wrong functional form.")

    # ---- 3. Heteroskedasticity ----
    section(3, "HETEROSKEDASTICITY")
    bp = het_breuschpagan(ols.resid, X)
    wh = het_white(ols.resid, X)
    print(f"  Breusch-Pagan   LM = {bp[0]:8.2f}, p = {bp[1]:.4f}")
    print(f"  White           LM = {wh[0]:8.2f}, p = {wh[1]:.4f}")
    print("\n  Rejection is expected here: county sizes vary by orders of")
    print("  magnitude. The paper reports HC3 standard errors, which are")
    print("  valid under heteroskedasticity, so this does not affect inference.")

    # ---- 4. Normality ----
    section(4, "NORMALITY OF ERRORS — Jarque-Bera")
    jb = jarque_bera(ols.resid)
    print(f"  JB = {jb[0]:.2f}, p = {jb[1]:.4f}")
    print(f"  skew = {jb[2]:.3f}, kurtosis = {jb[3]:.3f}")
    print(f"\n  Normality is not required for unbiasedness, and at n = "
          f"{int(ols.nobs)}")
    print("  the central limit theorem covers inference. Heavy tails matter")
    print("  only if a few observations drive the result, which check 5 tests.")

    # ---- 5. Influence ----
    section(5, "INFLUENTIAL OBSERVATIONS")
    infl = OLSInfluence(ols)
    cooks = infl.cooks_distance[0]
    leverage = infl.hat_matrix_diag
    k = X.shape[1]
    n = int(ols.nobs)

    print(f"  Max Cook's distance: {cooks.max():.4f}")
    print(f"  Counties above 4/n ({4 / n:.5f}): {(cooks > 4 / n).sum()}")
    print(f"  Counties above 1.0 (the usual alarm): {(cooks > 1).sum()}")
    print(f"  High leverage, above 2k/n ({2 * k / n:.4f}): "
          f"{(leverage > 2 * k / n).sum()}")

    keep = cooks <= 4 / n
    refit = sm.OLS(y[keep], X[keep]).fit(cov_type="HC3")
    print(f"\n  d_bb with all counties:        {hc3.params['d_bb']:.4f} "
          f"({hc3.bse['d_bb']:.4f})")
    print(f"  d_bb dropping high-influence:  {refit.params['d_bb']:.4f} "
          f"({refit.bse['d_bb']:.4f})   n = {int(refit.nobs)}")
    print("\n  If these two are close, no small group of counties is driving")
    print("  the result. This is the check that matters most given the heavy")
    print("  tails found above.")

    # ---- 6. Spatial dependence ----
    section(6, "SPATIAL DEPENDENCE — state-clustered standard errors")
    states = df["fips"].str[:2]
    clustered = sm.OLS(y, X).fit(cov_type="cluster",
                                 cov_kwds={"groups": states})
    print(f"  Clusters (states): {states.nunique()}")
    print(f"\n  {'':16s}{'HC3':>12s}{'Clustered':>12s}")
    for name in FULL:
        print(f"  {name:16s}{hc3.bse[name]:>12.4f}{clustered.bse[name]:>12.4f}")
    print(f"\n  d_bb p-value: HC3 {hc3.pvalues['d_bb']:.4f}, "
          f"clustered {clustered.pvalues['d_bb']:.4f}")
    print("\n  Counties in the same state share regional shocks, which makes")
    print("  errors correlated within states and HC3 standard errors too")
    print("  small. If the clustered p-value stays below 0.05, the result")
    print("  holds under the more conservative assumption.")

    print(f"\n{'=' * 70}")
    print("The two results that bear on the paper are check 5 (does a small")
    print("group of counties drive the estimate) and check 6 (does the result")
    print("survive clustering). The rest describe the data rather than")
    print("threaten the inference.")


if __name__ == "__main__":
    main()