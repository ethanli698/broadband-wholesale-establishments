import os
import io
import sys
import glob
import requests
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")

BASE = "https://www2.census.gov/programs-surveys/popest/datasets"
FILES = [
    (f"{BASE}/2000-2010/intercensal/county/co-est00int-tot.csv",
     {"POPESTIMATE2008": "pop2008"}, "2000-2010 intercensal"),
    (f"{BASE}/2010-2019/counties/totals/co-est2019-alldata.csv",
     {"POPESTIMATE2017": "pop2017", "POPESTIMATE2019": "pop2019"},
     "2010-2019 vintage"),
    (f"{BASE}/2020-2023/counties/totals/co-est2023-alldata.csv",
     {"POPESTIMATE2023": "pop2023"}, "2020-2023 vintage"),
]

OUT_FILE = os.path.join(HERE, "controls.csv")

SEARCH_DIRS = [HERE, os.path.dirname(HERE),
               os.path.join(HOME, "Downloads"), os.path.join(HOME, "Desktop")]
RUCC_PATTERNS = ["*ural*rban*2013*.xls*", "*ucc*2013*.xls*", "*2013*ural*.xls*"]


def fetch_population(url, colmap, label):
    """Download one PEP flat file and pull the requested year columns."""
    print(f"downloading {label} ...")
    r = requests.get(url, timeout=180)
    if r.status_code != 200:
        print(f"  FAILED status {r.status_code}\n  {url}")
        sys.exit(1)

    df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1", dtype=str)

    for src in colmap:
        if src not in df.columns:
            print(f"  '{src}' not in file. Population columns present:")
            print("  " + ", ".join(c for c in df.columns if "POP" in c.upper()))
            sys.exit(1)

    # Summary level 50 is the county level. Padding varies between files.
    df = df[pd.to_numeric(df["SUMLEV"], errors="coerce") == 50].copy()
    df["fips"] = df["STATE"].str.zfill(2) + df["COUNTY"].str.zfill(3)

    for src, dest in colmap.items():
        df[dest] = pd.to_numeric(df[src], errors="coerce")

    out = df[["fips"] + list(colmap.values())].drop_duplicates(subset="fips")
    for dest in colmap.values():
        print(f"  {dest}: {out[dest].notna().sum()} counties, "
              f"total {out[dest].sum():,.0f}")
    return out


def fetch_rucc():
    """Locate and read the USDA rural-urban codes spreadsheet."""
    path = None
    for folder in SEARCH_DIRS:
        if not os.path.isdir(folder):
            continue
        for pattern in RUCC_PATTERNS:
            hits = sorted(glob.glob(os.path.join(folder, pattern)))
            if hits:
                path = hits[0]
                break
        if path:
            break

    if path is None:
        print("\nCould not find the USDA rural-urban codes file. Looked in:")
        for folder in SEARCH_DIRS:
            print(f"    {folder}")
        print("\nDownload the 2013 edition from")
        print("    https://www.ers.usda.gov/data-products/"
              "rural-urban-continuum-codes")
        sys.exit(1)

    print(f"reading {os.path.basename(path)}")
    df = pd.read_excel(path, dtype=str)

    cols = {c.lower().strip(): c for c in df.columns}
    try:
        fips_col = next(c for k, c in cols.items() if k.startswith("fips"))
        rucc_col = next(c for k, c in cols.items() if "rucc" in k)
    except StopIteration:
        print(f"  Unexpected columns: {list(df.columns)}")
        sys.exit(1)

    out = pd.DataFrame({
        "fips": df[fips_col].astype(str).str.strip().str.zfill(5),
        "rucc": pd.to_numeric(df[rucc_col], errors="coerce"),
    }).dropna(subset=["rucc"])

    out["rucc"] = out["rucc"].astype(int)
    out["metro"] = (out["rucc"] <= 3).astype(int)

    print(f"  rucc: {len(out)} counties, {out['metro'].sum()} metro, "
          f"{len(out) - int(out['metro'].sum())} nonmetro")
    return out


def main():
    frames = [fetch_population(url, colmap, label)
              for url, colmap, label in FILES]

    pop = frames[0]
    for frame in frames[1:]:
        pop = pop.merge(frame, on="fips", how="outer")

    rucc = fetch_rucc()
    df = pop.merge(rucc, on="fips", how="outer", indicator=True)

    print("\nmerge:")
    print(df["_merge"].value_counts().to_string())

    missing = sorted(df.loc[df["_merge"] == "left_only", "fips"])
    if missing:
        print(f"\n{len(missing)} counties with population but no RUCC.")
        print("Expect Connecticut (09xxx planning regions, created 2022),")
        print("Alaska 02063/02066 (Valdez-Cordova split, 2019),")
        print("02158 and 46102 (renamed 2015). RUCC 2013 predates all four.")
        print("  " + ", ".join(missing[:30]))

    df = df.drop(columns="_merge")
    df.to_csv(OUT_FILE, index=False)

    needed = ["pop2008", "pop2017", "pop2019", "pop2023", "rucc"]
    complete = df.dropna(subset=needed)
    print(f"\nwrote {OUT_FILE}")
    print(f"  {len(df)} rows, {len(complete)} with all control variables")


if __name__ == "__main__":
    main()