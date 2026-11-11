import os
import time
import numpy as np
import polars as pl

PATH_IN = "data/raw/rna_seq_human/GTEx/GTEx.gct"
PATH_OUT = "data/processed/mRNA/human/GTX_TPM.csv"
WRITE_STEP = 2000   # rows per CSV write batch

t0 = time.time()
acc = {}   # Gene_ID (version stripped) -> float64 vector; duplicates (e.g. PAR_Y) get summed

with open(PATH_IN) as f:
    f.readline()                      # "#1.2"
    f.readline()                      # dims line
    header = f.readline().rstrip("\n").split("\t")
    samples = header[2:]              # skip Name, Description
    print(f"{len(samples)} samples", flush=True)

    for n, line in enumerate(f):
        fields = line.rstrip("\n").split("\t")
        gid = fields[0].split(".")[0]
        vals = np.array(fields[2:], dtype=np.float64)
        if gid in acc:
            acc[gid] += vals
        else:
            acc[gid] = vals
        if n % 2000 == 0:
            print(f"[{time.time() - t0:6.0f}s] row {n}", flush=True)

genes = sorted(acc)
mat = np.empty((len(genes), len(samples)), dtype=np.float32)
for i, g in enumerate(genes):
    mat[i] = np.log1p(acc.pop(g))     # sum first, then log1p, same as your original order
print(f"[{time.time() - t0:6.0f}s] matrix built: {mat.shape}", flush=True)

df = pl.from_numpy(mat, schema=samples, orient="row").insert_column(
    0, pl.Series("Gene_ID", genes)
)

os.makedirs(os.path.dirname(PATH_OUT), exist_ok=True)
with open(PATH_OUT, "wb") as out:
    for i in range(0, df.height, WRITE_STEP):
        df.slice(i, WRITE_STEP).write_csv(out, include_header=(i == 0), float_precision=3)
        print(f"[{time.time() - t0:6.0f}s] wrote {min(i + WRITE_STEP, df.height)}/{df.height}", flush=True)