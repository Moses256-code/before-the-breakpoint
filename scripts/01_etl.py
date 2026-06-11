"""
Phase 1 ETL: ATLAS wide CSV -> long parquet with parsed MIC intervals.
Writes each drug's chunk incrementally to avoid the OOM concat.
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from mic_parser import parse_series

ROOT = Path("/home/claude/atlas")
SRC  = Path("/home/claude/atlas_vivli_2004_2024_nonUSA.csv")
OUT_ISOLATES = ROOT / "data/atlas_isolates.parquet"
OUT_LONG     = ROOT / "data/atlas_long.parquet"

META_COLS = ["Isolate Id","Study","Species","Country","State","Gender","Age Group",
             "Speciality","Source","Year"]
GENOTYPE_COLS = ["ACC","ACT/MIR","AMPC","CMY I/MOX","CMY2","CTX-M-1","CTX-M-2",
                 "CTX-M-8/25","CTX-M-9","DHA","FOX","GES","GIM","IMP","KPC","NDM",
                 "OXA","PER","SHV","SPM","TEM","VEB","VIM"]
FOCUS_DRUGS = ["Amikacin","Amoxycillin clavulanate","Ampicillin","Ampicillin sulbactam",
               "Aztreonam","Aztreonam avibactam","Cefepime","Cefiderocol",
               "Cefoperazone sulbactam","Cefoxitin","Cefpodoxime","Ceftaroline",
               "Ceftazidime","Ceftazidime avibactam","Ceftolozane tazobactam",
               "Ceftriaxone","Ciprofloxacin","Colistin","Doripenem","Ertapenem",
               "Gentamicin","Imipenem","Levofloxacin","Meropenem",
               "Meropenem vaborbactam","Minocycline","Piperacillin tazobactam",
               "Tetracycline","Tigecycline","Trimethoprim sulfa"]

if not OUT_ISOLATES.exists():
    print("Building isolates table ...")
    iso_cols = META_COLS + GENOTYPE_COLS
    iso = pd.read_csv(SRC, usecols=iso_cols, dtype=str, low_memory=False)
    iso["Year"] = pd.to_numeric(iso["Year"], errors="coerce").astype("Int64")
    iso.to_parquet(OUT_ISOLATES, index=False)
    print(f"  wrote {OUT_ISOLATES}")

print("\nStreaming MIC parsing -> long parquet ...")
if OUT_LONG.exists():
    OUT_LONG.unlink()

schema = pa.schema([
    ("Isolate Id", pa.string()),
    ("Species",    pa.string()),
    ("Country",    pa.string()),
    ("Year",       pa.int16()),
    ("Age Group",  pa.string()),
    ("Source",     pa.string()),
    ("Speciality", pa.string()),
    ("Drug",       pa.string()),
    ("Interp",     pa.string()),
    ("mic_obs",    pa.string()),
    ("mic_label",  pa.float32()),
    ("log2_idx",   pa.int16()),
    ("log2_lower", pa.float32()),
    ("log2_upper", pa.float32()),
    ("cens_type",  pa.string()),
])

writer = pq.ParquetWriter(OUT_LONG, schema, compression="zstd")
total = 0
for i, drug in enumerate(FOCUS_DRUGS, 1):
    sub = pd.read_csv(SRC,
                      usecols=["Isolate Id","Species","Country","Year","Age Group","Source","Speciality",
                               drug, f"{drug}_I"],
                      dtype=str, low_memory=False)
    sub = sub[sub[drug].notna()]
    if not len(sub):
        print(f"  [{i:>2}/{len(FOCUS_DRUGS)}] {drug:<28} -> 0")
        continue
    parsed = parse_series(sub[drug])
    rec = pd.DataFrame({
        "Isolate Id": sub["Isolate Id"].values,
        "Species":    sub["Species"].values,
        "Country":    sub["Country"].values,
        "Year":       pd.to_numeric(sub["Year"], errors="coerce").astype("Int16").values,
        "Age Group":  sub["Age Group"].values,
        "Source":     sub["Source"].values,
        "Speciality": sub["Speciality"].values,
        "Drug":       drug,
        "Interp":     sub[f"{drug}_I"].values,
    })
    rec = pd.concat([rec.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    rec = rec[rec["log2_idx"].notna()].copy()
    for c in ["log2_lower","log2_upper"]:
        rec[c] = rec[c].astype("float32")
    rec["log2_idx"] = rec["log2_idx"].astype("int16")
    table = pa.Table.from_pandas(rec[ [f.name for f in schema] ], schema=schema, preserve_index=False)
    writer.write_table(table)
    n = len(rec); total += n
    print(f"  [{i:>2}/{len(FOCUS_DRUGS)}] {drug:<28} -> {n:>7,}")
    del sub, rec, parsed, table

writer.close()
print(f"\nDone. Wrote {total:,} long rows to {OUT_LONG}")

t = pq.read_table(OUT_LONG, columns=["Drug","cens_type"])
print("Verified row count:", t.num_rows)
df_ck = t.to_pandas()
print("Censoring overall:", df_ck["cens_type"].value_counts().to_dict())
