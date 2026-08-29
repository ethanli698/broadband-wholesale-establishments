import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))

BB = ["bb200", "bb10"]
BB_LABEL = {"bb200": "200 Kbps", "bb10": "10 Mbps"}

GROUPS = ["brokers", "durable", "nondurable"]
GROUP_LABEL = {
    "brokers": "Agents and brokers",
    "durable": "Merchant wholesalers, durable",
    "nondurable": "Merchant wholesalers, nondurable",
}

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


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def positive_only(df, cols):
    for col in cols:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan
    return df


def fit(df, outcome, regressors):
    sub = df.dropna(subset=[outcome] + regressors)
    X = sm.add_constant(sub[regressors], has_constant="add")
    return sm.OLS(sub[outcome], X).fit(cov_type="HC3")


def cell(m, var):
    if var not in m.params.index:
        return ["", "", ""]
    return [f"{m.params[var]:.4f}{stars(m.pvalues[var])}",
            f"({m.bse[var]:.4f})",
            f"[{m.pvalues[var]:.3f}]"]


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load():
    panel = pd.read_csv(find("wholesale_panel.csv"), dtype={"fips": str})
    panel["fips"] = panel["fips"].str.strip().str.zfill(5)

    controls = pd.read_csv(find("controls.csv"), dtype={"fips": str})
    controls["fips"] = controls["fips"].str.strip().str.zfill(5)

    for col in ["pop2008", "pop2017", "pop2019", "pop2023"]:
        if col not in controls.columns:
            print(f"\ncontrols.csv is missing {col}. "
                  f"Rerun 01_build_controls.py.")
            sys.exit(1)

    groups = list(panel["group"].unique())
    aggregate = next((g for g in groups
                      if str(g).lower() in ("all", "total", "wholesale", "42")),
                     None)
    if aggregate is None:
        print(f"\nNo aggregate group found. Groups present: {groups}")
        sys.exit(1)

    print(f"panel: {len(panel)} rows, groups {groups}")
    print(f"aggregate group: '{aggregate}'")
    print(f"controls: {len(controls)} rows")

    return panel, controls, aggregate


def build(panel, controls, group, start, end, bb):
    df = panel[panel["group"] == group].copy()
    df = df.merge(controls, on="fips", how="left")

    df = positive_only(df, [f"estab_{start}", f"estab_{end}",
                            f"pop{start}", f"pop{end}"])

    if MIN_ESTAB > 0:
        before = int(df[f"estab_{start}"].notna().sum())
        df = df[df[f"estab_{start}"] >= MIN_ESTAB].copy()
        dropped = before - len(df)
        if dropped:
            print(f"    [{group}, {start}-{end}] dropped {dropped} counties "
                  f"with fewer than {MIN_ESTAB} establishments in {start}")

    df["d_lestab"] = np.log(df[f"estab_{end}"]) - np.log(df[f"estab_{start}"])
    df["lestab_base"] = np.log(df[f"estab_{start}"])
    df["d_lpop"] = np.log(df[f"pop{end}"]) - np.log(df[f"pop{start}"])

    if bb == "acs":
        df["d_bb"] = df[f"acs_{end}"] - df[f"acs_{start}"]
        df["bb_base"] = df[f"acs_{start}"]
    else:
        df["d_bb"] = df[f"{bb}_{end}"] - df[f"{bb}_{start}"]
        df["bb_base"] = df[f"{bb}_{start}"]

    if f"emp_{start}" in df.columns and f"emp_{end}" in df.columns:
        df["size_base"] = df[f"emp_{start}"] / df[f"estab_{start}"]
        df["size_end"] = df[f"emp_{end}"] / df[f"estab_{end}"]
        df = positive_only(df, ["size_base", "size_end"])
        df["d_lsize"] = np.log(df["size_end"]) - np.log(df["size_base"])

    return df


def table2(panel, controls, aggregate):
    header("TABLE 2 — SUMMARY STATISTICS, 2008-2023")

    df = build(panel, controls, aggregate, 2008, 2023, "bb200")
    df10 = build(panel, controls, aggregate, 2008, 2023, "bb10")
    df["d_bb10"] = df10["d_bb"]
    df["bb10_base"] = df10["bb_base"]

    rows = [
        ("Change in broadband tier, 200 Kbps", "d_bb"),
        ("Broadband tier in 2008, 200 Kbps", "bb_base"),
        ("Change in broadband tier, 10 Mbps", "d_bb10"),
        ("Broadband tier in 2008, 10 Mbps", "bb10_base"),
        ("Wholesale establishments, 2008", "estab_2008"),
        ("Wholesale establishments, 2023", "estab_2023"),
        ("Average establishment size, 2008", "size_base"),
        ("Average establishment size, 2023", "size_end"),
        ("Change in log establishments", "d_lestab"),
        ("Change in log average size", "d_lsize"),
        ("Change in log population", "d_lpop"),
        ("Log establishments, 2008", "lestab_base"),
        ("Metro county", "metro"),
    ]

    print(f"{'Variable':38s}{'Mean':>10s}{'SD':>10s}"
          f"{'Median':>10s}{'Min':>10s}{'Max':>10s}")
    print("-" * 78)
    for label, col in rows:
        s = df[col].dropna()
        if col == "metro":
            print(f"{label:38s}{s.mean():>10.3f}{'':>10s}"
                  f"{'':>10s}{s.min():>10.0f}{s.max():>10.0f}")
        else:
            print(f"{label:38s}{s.mean():>10.3f}{s.std():>10.3f}"
                  f"{s.median():>10.3f}{s.min():>10.3f}{s.max():>10.3f}")

    print(f"\nEstablishment totals: 2008 {df['estab_2008'].sum():,.0f}, "
          f"2023 {df['estab_2023'].sum():,.0f}")
    change = df["estab_2023"].sum() / df["estab_2008"].sum() - 1
    print(f"Net change: {change:.1%}")
    print("Note: the metro row reports a share; SD and median are omitted "
          "for an indicator.")


SPECS = [
    ("(1)", ["d_bb"]),
    ("(2)", ["d_bb", "bb_base"]),
    ("(3)", ["d_bb", "bb_base", "lestab_base"]),
    ("(4)", ["d_bb", "bb_base", "lestab_base", "d_lpop"]),
    ("(5)", ["d_bb", "bb_base", "lestab_base", "d_lpop", "metro"]),
]

VAR_LABEL = {
    "d_bb": "Change in broadband tier",
    "bb_base": "Broadband tier, 2008",
    "lestab_base": "Log establishments, 2008",
    "d_lpop": "Change in log population",
    "metro": "Metro county",
}


def buildup(df, outcome, show_controls):
    models = [(name, fit(df, outcome, regs)) for name, regs in SPECS]

    print(f"{'':30s}" + "".join(f"{n:>13s}" for n, _ in models))
    print("-" * (30 + 13 * len(models)))

    variables = list(VAR_LABEL) if show_controls else ["d_bb"]
    for var in variables:
        lines = [cell(m, var) for _, m in models]
        if all(line[0] == "" for line in lines):
            continue
        print(f"{VAR_LABEL[var]:30s}" + "".join(f"{l[0]:>13s}" for l in lines))
        print(f"{'':30s}" + "".join(f"{l[1]:>13s}" for l in lines))
        if var == "d_bb":
            print(f"{'':30s}" + "".join(f"{l[2]:>13s}" for l in lines))

    print(f"{'R-squared':30s}" +
          "".join(f"{m.rsquared:>13.4f}" for _, m in models))
    print(f"{'Observations':30s}" +
          "".join(f"{int(m.nobs):>13d}" for _, m in models))


def table3(panel, controls, aggregate):
    header("TABLE 3 — BROADBAND AND WHOLESALE ESTABLISHMENTS, 2008-2023")

    for outcome, panel_name, show in [
            ("d_lestab", "Panel A. Log total establishments", True),
            ("d_lsize", "Panel B. Log average establishment size", False)]:
        print(f"\n{panel_name}")
        for bb in BB:
            print(f"\n  {BB_LABEL[bb]}")
            df = build(panel, controls, aggregate, 2008, 2023, bb)
            buildup(df, outcome, show)


FULL = ["d_bb", "bb_base", "lestab_base", "d_lpop", "metro"]


def pair(df, outcome):
    return fit(df, outcome, ["d_bb"]), fit(df, outcome, FULL)


def print_pairs(rows):
    print(f"{'':34s}{'Bivariate':>19s}{'With controls':>19s}")
    print("-" * 72)
    for label, m0, m1 in rows:
        c0, c1 = cell(m0, "d_bb"), cell(m1, "d_bb")
        print(f"{label:34s}{c0[0]:>19s}{c1[0]:>19s}")
        print(f"{'':34s}{c0[1]:>19s}{c1[1]:>19s}")
        print(f"{'':34s}{c0[2]:>19s}{c1[2]:>19s}")
        print(f"{'R-squared':34s}{m0.rsquared:>19.4f}{m1.rsquared:>19.4f}")
        print(f"{'Observations':34s}{int(m0.nobs):>19d}{int(m1.nobs):>19d}")
        print()


def table4(panel, controls):
    header("TABLE 4 — ESTABLISHMENT GROWTH BY INDUSTRY GROUP, 2008-2023")
    for bb in BB:
        print(f"\n{BB_LABEL[bb]}")
        rows = []
        for g in GROUPS:
            if g not in panel["group"].values:
                print(f"  group '{g}' not in panel")
                continue
            df = build(panel, controls, g, 2008, 2023, bb)
            rows.append((GROUP_LABEL[g], *pair(df, "d_lestab")))
        print_pairs(rows)


def table5(panel, controls, aggregate):
    header("TABLE 5 — PRE-PANDEMIC WINDOW, 2008-2019")
    rows = []
    for bb in BB:
        df = build(panel, controls, aggregate, 2008, 2019, bb)
        rows.append((BB_LABEL[bb], *pair(df, "d_lestab")))
    print_pairs(rows)


def table6(panel, controls, aggregate):
    header("TABLE 6 — AMERICAN COMMUNITY SURVEY PANEL, 2017-2023")
    df = build(panel, controls, aggregate, 2017, 2023, "acs")
    rows = [("Log total establishments", *pair(df, "d_lestab")),
            ("Log average size", *pair(df, "d_lsize"))]
    print_pairs(rows)
    print("Note: broadband adoption is the ACS household subscription share, "
          "so the\ncoefficient is per unit of share rather than per tier.")


def main():
    panel, controls, aggregate = load()
    table2(panel, controls, aggregate)
    table3(panel, controls, aggregate)
    table4(panel, controls)
    table5(panel, controls, aggregate)
    table6(panel, controls, aggregate)


if __name__ == "__main__":
    main()