import re
import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import pandas as pd

UNIPROT_RE = re.compile(r"uniprot/([A-Z0-9]+)")


def get_uniprot_id(jaspar_id):
    url = f"https://jaspar.elixir.no/matrix/{jaspar_id}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.find_all("a", href=True):
        match = UNIPROT_RE.search(a["href"])
        if match:
            return match.group(1)

    return None  # in case no UniProt link exists


def extract_jaspar_ids(meme_file):
    ids = []
    with open(meme_file) as f:
        for line in f:
            if line.startswith("MOTIF"):
                _, jaspar_id, *_ = line.split()
                ids.append(jaspar_id)
    return ids


def build_mapping(meme_file, output_file):
    jaspar_ids = extract_jaspar_ids(meme_file)

    with open(output_file, "w") as out:
        for jid in tqdm(jaspar_ids, desc="Fetching UniProt IDs"):
            try:
                uniprot = get_uniprot_id(jid)
                out.write(f"{jid}\t{uniprot or 'NA'}\n")
                # time.sleep(0.05)  # be polite to JASPAR
            except Exception:
                out.write(f"{jid}\tERROR\n")


def substitute_jaspar_with_uniprot(meme_in, meme_out, mapping_txt):
    """
    Parameters
    ----------
    meme_in : str
        Input MEME file
    meme_out : str
        Output MEME file
    mapping_txt : str
        TSV file: JASPAR_ID <tab> UniProt_ID
    """

    # ---- step 1: load mapping into dict via pandas ----
    df = pd.read_csv(mapping_txt, sep="\t", header=None, names=["jaspar", "uniprot"])

    jaspar_to_uniprot = (
        df.assign(uniprot=lambda d: d["uniprot"].replace({"NA": pd.NA}))
        .dropna(subset=["uniprot"])
        .set_index("jaspar")["uniprot"]
        .to_dict()
    )

    # ---- step 2: rewrite MEME file ----
    with open(meme_in) as fin, open(meme_out, "w") as fout:
        for line in tqdm(fin, desc="Rewriting MEME file"):
            if line.startswith("MOTIF"):
                parts = line.rstrip("\n").split()
                if len(parts) >= 3:
                    jaspar_id = parts[1]
                    if jaspar_id in jaspar_to_uniprot:
                        parts[1] = jaspar_to_uniprot[jaspar_id]
                        line = " ".join(parts) + "\n"

            fout.write(line)


if __name__ == "__main__":

    build_mapping(
        "data/processed/regulation/human/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt",
        "data/processed/regulation/human/JASPAR-UNIPROT.txt",
    )
    substitute_jaspar_with_uniprot(
        "data/processed/regulation/human/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt",
        "data/processed/regulation/human/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme_uniprot.txt",
        "data/processed/regulation/human/JASPAR-UNIPROT.txt",
    )

    build_mapping(
        "data/processed/regulation/ath/JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt",
        "data/processed/regulation/ath/JASPAR-UNIPROT.txt",
    )
    substitute_jaspar_with_uniprot(
        "data/processed/regulation/ath/JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt",
        "data/processed/regulation/ath/JASPAR2026_CORE_plants_non-redundant_pfms_meme_uniprot.txt",
        "data/processed/regulation/ath/JASPAR-UNIPROT.txt",
    )
