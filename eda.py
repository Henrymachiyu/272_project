#!/usr/bin/env python3
"""EDA for the presentation: split / OOS shift, lengths, vocabulary, label-name
cues, confusable intents and how close OOS inputs are to supported intents.

    python eda.py            # writes results/eda/{eda_stats.json, eda_summary.md, fig_*.png}
"""

import json
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "eda")

# CLINC150 domains (Larson et al., 2019), 15 intents each.
DOMAINS = {
    "banking": "transfer transactions balance freeze_account pay_bill bill_balance bill_due interest_rate routing "
               "min_payment order_checks pin_change report_fraud account_blocked spending_history",
    "credit cards": "credit_score report_lost_card credit_limit rewards_balance new_card application_status "
                    "card_declined international_fees apr redeem_rewards credit_limit_change damaged_card "
                    "replacement_card_duration improve_credit_score expiration_date",
    "kitchen & dining": "recipe restaurant_reviews calories nutrition_info restaurant_suggestion ingredients_list "
                        "ingredient_substitution cook_time food_last meal_suggestion restaurant_reservation "
                        "confirm_reservation how_busy cancel_reservation accept_reservations",
    "home": "shopping_list shopping_list_update next_song play_music update_playlist todo_list todo_list_update "
            "calendar calendar_update what_song order order_status reminder reminder_update smart_home",
    "auto & commute": "traffic directions gas gas_type distance current_location mpg oil_change_when oil_change_how "
                      "jump_start uber schedule_maintenance last_maintenance tire_pressure tire_change",
    "travel": "book_flight book_hotel car_rental travel_suggestion travel_alert travel_notification carry_on "
              "timezone vaccines translate flight_status international_visa lost_luggage plug_type exchange_rate",
    "utility": "time alarm share_location find_phone weather text spelling make_call timer date calculator "
               "measurement_conversion flip_coin roll_dice definition",
    "work": "direct_deposit pto_request taxes payday w2 pto_balance pto_request_status next_holiday insurance "
            "insurance_change schedule_meeting pto_used meeting_schedule rollover_401k income",
    "small talk": "greeting goodbye tell_joke where_are_you_from how_old_are_you what_is_your_name who_made_you "
                  "thank_you what_can_i_ask_you what_are_your_hobbies do_you_have_pets are_you_a_bot "
                  "meaning_of_life who_do_you_work_for fun_fact",
    "meta": "change_ai_name change_user_name cancel user_name reset_settings whisper_mode repeat no yes maybe "
            "change_language change_accent change_volume change_speed sync_device",
}
DOMAIN_OF = {i: d for d, s in DOMAINS.items() for i in s.split()}

STOP = set("a an the to of in on for my me i is it what how do does can you your be at and or with this that "
           "from by are am was will should would please tell know".split())

# palette (dataviz reference instance, light mode)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def tok(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def main():
    D = json.load(open(os.path.join(HERE, "data", "data_oos_plus.json")))
    S = {s: D[s] + D["oos_" + s] for s in ("train", "val", "test")}
    intents = sorted({l for _, l in D["train"]})
    assert set(DOMAIN_OF) == set(intents) and len(DOMAIN_OF) == 150, set(intents) ^ set(DOMAIN_OF)
    os.makedirs(OUT, exist_ok=True)
    st = {}

    # 1. split sizes and OOS share --------------------------------------------------------
    st["splits"] = {s: {"total": len(r), "in_scope": sum(l != "oos" for _, l in r),
                        "oos": sum(l == "oos" for _, l in r)} for s, r in S.items()}
    for v in st["splits"].values():
        v["oos_pct"] = round(100 * v["oos"] / v["total"], 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    names = list(S)
    ins = [st["splits"][s]["in_scope"] for s in names]
    oos = [st["splits"][s]["oos"] for s in names]
    ax[0].bar(names, ins, color=BLUE, width=0.55, label="in-scope")
    ax[0].bar(names, oos, bottom=ins, color=ORANGE, width=0.55, label="OOS", edgecolor=SURF, linewidth=2)
    for i, s in enumerate(names):
        ax[0].text(i, ins[i] + oos[i] + 250, f"{ins[i] + oos[i]:,}", ha="center", color=INK2)
    ax[0].set_title("Split sizes")
    ax[0].set_ylim(0, 17500)
    ax[0].legend(loc="upper right")
    pct = [st["splits"][s]["oos_pct"] for s in names]
    ax[1].bar(names, pct, color=ORANGE, width=0.55)
    for i, p in enumerate(pct):
        ax[1].text(i, p + 0.5, f"{p}%", ha="center", color=INK, fontweight="bold")
        ax[1].text(i, -2.2, f"n={oos[i]:,}", ha="center", color=MUTED, fontsize=10)
    ax[1].set_title("OOS share per split")
    ax[1].set_ylim(0, 21)
    ax[1].set_yticks([0, 5, 10, 15, 20])
    ax[1].set_yticklabels([f"{t}%" for t in [0, 5, 10, 15, 20]])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_splits.png"), dpi=200)
    plt.close(fig)

    # 2. lengths ---------------------------------------------------------------------------
    allr = sum(S.values(), [])
    L_in = np.array([len(t.split()) for t, l in allr if l != "oos"])
    L_oos = np.array([len(t.split()) for t, l in allr if l == "oos"])
    q = lambda a: {"mean": round(float(a.mean()), 2), "median": float(np.median(a)),
                   "p95": float(np.percentile(a, 95)), "max": int(a.max()), "min": int(a.min())}
    st["length_words"] = {"in_scope": q(L_in), "oos": q(L_oos)}
    st["length_by_domain"] = {d: round(float(np.mean([len(t.split()) for t, l in allr if DOMAIN_OF.get(l) == d])), 2)
                              for d in DOMAINS}
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0.5, 29.5, 1)
    ax.hist(L_in, bins=bins, density=True, color=BLUE, alpha=0.85, label=f"in-scope (mean {L_in.mean():.1f})",
            edgecolor=SURF, linewidth=1)
    ax.hist(L_oos, bins=bins, density=True, histtype="step", color=ORANGE, linewidth=2.2,
            label=f"OOS (mean {L_oos.mean():.1f})")
    ax.set_xlabel("query length (words)")
    ax.set_ylabel("share of queries")
    ax.set_title("Query length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_lengths.png"), dpi=200)
    plt.close(fig)

    # 3. data hygiene, cross-split leakage ----------------------------------------------------
    norm = lambda t: " ".join(tok(t))
    tr = {norm(t) for t, _ in S["train"]}
    st["hygiene"] = {
        "empty": sum(not t.strip() for t, _ in allr),
        "dup_rows_text_label": len(allr) - len(set(allr if isinstance(allr[0], tuple) else map(tuple, allr))),
        "test_texts_seen_in_train": sum(norm(t) in tr for t, _ in S["test"]),
        "val_texts_seen_in_train": sum(norm(t) in tr for t, _ in S["val"]),
    }

    # 4. vocabulary / OOV -------------------------------------------------------------------
    V = Counter(w for t, _ in S["train"] for w in tok(t))
    def oov(rows):
        ws = [w for t, _ in rows for w in tok(t)]
        return round(100 * sum(w not in V for w in ws) / len(ws), 2)
    def utt_with_oov(rows):
        return round(100 * np.mean([any(w not in V for w in tok(t)) for t, _ in rows]), 1)
    tin = [r for r in S["test"] if r[1] != "oos"]
    toos = [r for r in S["test"] if r[1] == "oos"]
    st["vocab"] = {"train_vocab": len(V),
                   "oov_token_pct": {"test_in_scope": oov(tin), "test_oos": oov(toos)},
                   "utt_with_oov_pct": {"test_in_scope": utt_with_oov(tin), "test_oos": utt_with_oov(toos)}}

    # 5. label-name cue: does the utterance contain a word from its intent name? ------------
    def name_words(l):
        return {w for w in l.split("_") if w not in STOP and len(w) > 1}
    def has_cue(t, l):
        ws = set(tok(t))
        stems = {w.rstrip("s") for w in ws}
        return any(w in ws or w.rstrip("s") in stems for w in name_words(l))
    rows_in = [r for r in S["test"] if r[1] != "oos"]
    cue = {l: np.mean([has_cue(t, l) for t, ll in rows_in if ll == l]) for l in intents}
    st["label_name_cue"] = {
        "pct_test_in_scope_with_name_word": round(100 * np.mean([has_cue(t, l) for t, l in rows_in]), 1),
        "lowest_intents": sorted(((l, round(100 * v, 1)) for l, v in cue.items()), key=lambda x: x[1])[:10],
        "highest_intents": sorted(((l, round(100 * v, 1)) for l, v in cue.items()), key=lambda x: -x[1])[:5],
    }

    # 6. confusable intents: TF-IDF centroid cosine ----------------------------------------
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform([t for t, l in D["train"]])
    ytr = np.array([l for _, l in D["train"]])
    order = [i for d in DOMAINS for i in DOMAINS[d].split()]
    C = normalize(np.vstack([np.asarray(Xtr[ytr == l].mean(0)) for l in order]))
    Sim = C @ C.T
    iu = np.triu_indices(150, 1)
    pairs = sorted(zip(Sim[iu], iu[0], iu[1]), reverse=True)
    top = [(order[a], order[b], round(float(s), 3), DOMAIN_OF[order[a]] == DOMAIN_OF[order[b]])
           for s, a, b in pairs[:15]]
    st["confusable_pairs_top15"] = top
    same = np.array([DOMAIN_OF[order[a]] == DOMAIN_OF[order[b]] for a, b in zip(*iu)])
    st["centroid_sim"] = {"within_domain_mean": round(float(Sim[iu][same].mean()), 3),
                          "cross_domain_mean": round(float(Sim[iu][~same].mean()), 3),
                          "top50_pairs_within_domain": int(sum(DOMAIN_OF[order[a]] == DOMAIN_OF[order[b]]
                                                               for _, a, b in pairs[:50]))}

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    M = Sim.copy()
    np.fill_diagonal(M, np.nan)
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=0.6, interpolation="nearest")
    ax.grid(False)
    edges = np.arange(0, 151, 15)
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color=INK2, lw=0.5)
        ax.axvline(e - 0.5, color=INK2, lw=0.5)
    ax.set_xticks(edges[:-1] + 7)
    ax.set_xticklabels(list(DOMAINS), rotation=45, ha="right", fontsize=10, color=INK2)
    ax.set_yticks(edges[:-1] + 7)
    ax.set_yticklabels(list(DOMAINS), fontsize=10, color=INK2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("cosine similarity (TF-IDF centroids)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Intent similarity, grouped by domain")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_intent_similarity.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.6))
    tp = top[:12][::-1]
    ax.barh([f"{a}  /  {b}" for a, b, _, _ in tp], [s for _, _, s, _ in tp],
            color=[BLUE if w else ORANGE for *_, w in tp], height=0.6)
    for i, (_, _, s, _) in enumerate(tp):
        ax.text(s + 0.008, i, f"{s:.2f}", va="center", color=INK2, fontsize=10)
    ax.set_xlim(0, max(s for *_, s, _ in tp) + 0.1)
    ax.set_xlabel("cosine similarity (TF-IDF centroids)")
    ax.set_title("Most similar intent pairs")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="same domain"), Patch(color=ORANGE, label="different domains")],
              loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=2)
    ax.tick_params(axis="y", colors=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_confusable_pairs.png"), dpi=200)
    plt.close(fig)

    # 7. how close are OOS inputs to supported intents? -------------------------------------
    Xte = normalize(vec.transform([t for t, _ in S["test"]]))
    yte = np.array([l for _, l in S["test"]])
    simte = np.asarray(Xte @ C.T)
    mx = simte.max(1)
    nn = np.array(order)[simte.argmax(1)]
    is_oos = yte == "oos"
    st["oos_proximity"] = {
        "max_centroid_sim_mean": {"in_scope": round(float(mx[~is_oos].mean()), 3),
                                  "oos": round(float(mx[is_oos].mean()), 3)},
        "auroc_max_sim_as_inscope_score": round(100 * roc_auc_score(~is_oos, mx), 1),
        "oos_nearest_domain": Counter(DOMAIN_OF[n] for n in nn[is_oos]).most_common(),
        "oos_nearest_intent_top10": Counter(nn[is_oos]).most_common(10),
        "oos_examples_closest": [(S["test"][i][0], nn[i], round(float(mx[i]), 3))
                                 for i in np.where(is_oos)[0][np.argsort(-mx[is_oos])][:8]],
        "oos_examples_farthest": [(S["test"][i][0], nn[i], round(float(mx[i]), 3))
                                  for i in np.where(is_oos)[0][np.argsort(mx[is_oos])][:5]],
    }
    st["oos_proximity"]["pct_oos_above_median_inscope_sim"] = round(
        100 * float(np.mean(mx[is_oos] > np.median(mx[~is_oos]))), 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    b = np.linspace(0, 1, 41)
    ax[0].hist(mx[~is_oos], bins=b, weights=np.full((~is_oos).sum(), 100 / (~is_oos).sum()), color=BLUE, alpha=0.85, edgecolor=SURF, label="in-scope (test)")
    ax[0].hist(mx[is_oos], bins=b, weights=np.full(is_oos.sum(), 100 / is_oos.sum()), histtype="step", color=ORANGE, lw=2.2, label="OOS (test)")
    ax[0].set_xlabel("max similarity to an intent centroid")
    ax[0].set_ylabel("% of queries")
    ax[0].set_title("Similarity to the nearest intent")
    ax[0].legend()
    nd = st["oos_proximity"]["oos_nearest_domain"][::-1]
    ax[1].barh([d for d, _ in nd], [c / is_oos.sum() * 100 for _, c in nd], color=ORANGE, height=0.6)
    for i, (_, c) in enumerate(nd):
        ax[1].text(c / is_oos.sum() * 100 + 0.4, i, f"{100 * c / is_oos.sum():.0f}%", va="center",
                   color=INK2, fontsize=10)
    ax[1].set_xlabel("% of test OOS")
    ax[1].set_title("Nearest domain of OOS inputs")
    ax[1].grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_oos_proximity.png"), dpi=200)
    plt.close(fig)


    # 8. examples ----------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    st["examples"] = {
        "in_scope": [(t, l) for t, l in (S["train"][i] for i in rng.choice(15000, 6, replace=False))],
        "oos_train": [t for t, _ in D["oos_train"][:5]],
    }

    json.dump(st, open(os.path.join(OUT, "eda_stats.json"), "w"), indent=1, default=str)
    print(json.dumps(st, indent=1, default=str))


if __name__ == "__main__":
    main()
