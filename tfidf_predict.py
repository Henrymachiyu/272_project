#!/usr/bin/env python3
"""Predict intents with the saved TF-IDF models (run `tfidf_baseline.py --save-models` first).

    python tfidf_predict.py "how much has the dow changed today" "what's the spanish word for pasta"
    python tfidf_predict.py --test-head 10          # first N test queries, with gold labels

For each input prints the 151-label prediction (argmax, may be oos) and the 150-label
prediction with its MSP; the 150 model refuses (oos) when MSP < t from val.
"""

import argparse
import json
import os

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
DATA = os.path.join(HERE, "data", "data_oos_plus.json")
OOS = "oos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*")
    ap.add_argument("--test-head", type=int, default=0, help="also run the first N test queries")
    args = ap.parse_args()

    texts, gold = list(args.texts), [None] * len(args.texts)
    if args.test_head:
        D = json.load(open(DATA))
        rows = (D["test"] + D["oos_test"])[:args.test_head]
        texts += [t for t, _ in rows]
        gold += [lab for _, lab in rows]
    if not texts:
        ap.error("give some texts or --test-head N")

    m151 = joblib.load(os.path.join(MODELS, "tfidf_151.joblib"))["model"]
    b150 = joblib.load(os.path.join(MODELS, "tfidf_150.joblib"))
    m150, tau = b150["model"], b150["tau"]
    P, Q = m151.predict_proba(texts), m150.predict_proba(texts)
    oos_col = list(m151.classes_).index(OOS)

    print(f"150-label cutoff t = {tau:.2f}\n")
    print(f"{'text':48s} {'gold':16s} {'151: pred (P)':34s} {'150: pred (MSP)':30s} {'150 final':16s}")
    for i, t in enumerate(texts):
        p151 = m151.classes_[P[i].argmax()]
        p150, msp = m150.classes_[Q[i].argmax()], Q[i].max()
        final = p150 if msp >= tau else OOS
        print(f"{t[:47]:48s} {gold[i] or '-':16s} "
              f"{f'{p151} ({P[i].max():.3f}, P(oos)={P[i][oos_col]:.3f})':34s} "
              f"{f'{p150} ({msp:.3f})':30s} {final:16s}")


if __name__ == "__main__":
    main()
