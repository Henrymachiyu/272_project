#!/usr/bin/env python3
"""Download CLINC150 (OOS-plus) to data/data_oos_plus.json and check the split sizes.

We use the canonical JSON from https://github.com/clinc/oos-eval (CC BY 3.0), not the
HuggingFace `clinc_oos` mirror, which repeats every example three times.

    python download_data.py            # skips the download if the file is already there
    python download_data.py --force    # download again
"""

import argparse
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json"
PATH = os.path.join(HERE, "data", "data_oos_plus.json")
EXPECTED = {"train": 15000, "oos_train": 250, "val": 3000, "oos_val": 100,
            "test": 4500, "oos_test": 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="download even if the file exists")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    if args.force or not os.path.exists(PATH):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, PATH)
    D = json.load(open(PATH))

    sizes = {k: len(D[k]) for k in EXPECTED}
    assert sizes == EXPECTED, f"unexpected split sizes {sizes}"
    intents = {lab for _, lab in D["train"]}
    assert len(intents) == 150 and "oos" not in intents
    assert all(lab == "oos" for k in ("oos_train", "oos_val", "oos_test") for _, lab in D[k])

    print(f"ok: {PATH}")
    for s in ("train", "val", "test"):
        n_in, n_oos = sizes[s], sizes["oos_" + s]
        print(f"  {s:5s} total={n_in + n_oos:6,d}  in-scope={n_in:6,d}  oos={n_oos:5,d}")
    print(f"  {len(intents)} intents + oos = {len(intents) + 1} labels")


if __name__ == "__main__":
    main()
