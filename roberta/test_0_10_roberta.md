## RoBERTa-base 151 (seed 0, 5 epochs)
confidence score = top in-scope intent probability; refuse if argmax is `oos` or conf < tau = 0.17 (tau picked on val by OOS F1). For comparison with the 3B rows: 1 - p(oos) is >= 0.9988 on all 10 sentences.

| # | text | gold | argmax (no generation) | top-3 (p) | conf | final | ok |
|---|---|---|---|---|---|---|---|
| 0 | how would you say fly in italian | translate | `translate` | translate (0.986)<br>change_language (0.001)<br>spelling (0.001) | 0.9858<br>p(oos) 0.0001 | translate | ✓ |
| 1 | what's the spanish word for pasta | translate | `translate` | translate (0.984)<br>spelling (0.002)<br>change_language (0.001) | 0.9836<br>p(oos) 0.0003 | translate | ✓ |
| 2 | how would they say butter in zambia | translate | `translate` | translate (0.985)<br>spelling (0.001)<br>timezone (0.001) | 0.9848<br>p(oos) 0.0002 | translate | ✓ |
| 3 | how do you say fast in spanish | translate | `translate` | translate (0.986)<br>change_language (0.001)<br>spelling (0.001) | 0.9858<br>p(oos) 0.0001 | translate | ✓ |
| 4 | what's the word for trees in norway | translate | `translate` | translate (0.916)<br>spelling (0.025)<br>definition (0.014) | 0.9165<br>p(oos) 0.0012 | translate | ✓ |
| 5 | how does one say wonderful in german | translate | `translate` | translate (0.986)<br>spelling (0.001)<br>change_language (0.001) | 0.9856<br>p(oos) 0.0001 | translate | ✓ |
| 6 | how do they say tacos in mexico | translate | `translate` | translate (0.985)<br>spelling (0.001)<br>change_language (0.001) | 0.9855<br>p(oos) 0.0001 | translate | ✓ |
| 7 | how would one say cruiser in china | translate | `translate` | translate (0.986)<br>change_language (0.001)<br>spelling (0.001) | 0.9859<br>p(oos) 0.0001 | translate | ✓ |
| 8 | what's the french word you use for potato | translate | `translate` | translate (0.984)<br>change_language (0.001)<br>spelling (0.001) | 0.9839<br>p(oos) 0.0004 | translate | ✓ |
| 9 | what would the word for grass be in finland | translate | `translate` | translate (0.948)<br>spelling (0.018)<br>definition (0.004) | 0.9476<br>p(oos) 0.0006 | translate | ✓ |

## RoBERTa-base 150 + cutoff (seed 0, 5 epochs)
confidence score = top intent probability (150 classes, no oos class); refuse if conf < tau = 0.79 (tau picked on val by OOS F1)

| # | text | gold | argmax (no generation) | top-3 (p) | conf | final | ok |
|---|---|---|---|---|---|---|---|
| 0 | how would you say fly in italian | translate | `translate` | translate (0.988)<br>spelling (0.001)<br>change_language (0.001) | 0.9881 | translate | ✓ |
| 1 | what's the spanish word for pasta | translate | `translate` | translate (0.987)<br>spelling (0.001)<br>change_language (0.001) | 0.9868 | translate | ✓ |
| 2 | how would they say butter in zambia | translate | `translate` | translate (0.983)<br>spelling (0.002)<br>fun_fact (0.001) | 0.9829 | translate | ✓ |
| 3 | how do you say fast in spanish | translate | `translate` | translate (0.988)<br>change_language (0.001)<br>spelling (0.001) | 0.9878 | translate | ✓ |
| 4 | what's the word for trees in norway | translate | `translate` | translate (0.821)<br>definition (0.080)<br>spelling (0.026) | 0.8213 | translate | ✓ |
| 5 | how does one say wonderful in german | translate | `translate` | translate (0.988)<br>spelling (0.001)<br>change_language (0.001) | 0.9878 | translate | ✓ |
| 6 | how do they say tacos in mexico | translate | `translate` | translate (0.986)<br>spelling (0.001)<br>fun_fact (0.001) | 0.9857 | translate | ✓ |
| 7 | how would one say cruiser in china | translate | `translate` | translate (0.988)<br>spelling (0.001)<br>change_language (0.001) | 0.9878 | translate | ✓ |
| 8 | what's the french word you use for potato | translate | `translate` | translate (0.987)<br>change_language (0.002)<br>spelling (0.001) | 0.9868 | translate | ✓ |
| 9 | what would the word for grass be in finland | translate | `translate` | translate (0.958)<br>spelling (0.008)<br>definition (0.007) | 0.9580 | translate | ✓ |

## Seeds 1 and 2 (from saved test predictions)

| model | seed | tau | correct | rows below 0.9 conf |
|---|---|---|---|---|
| 151 | 1 | 0.00 | 10/10 | #4 translate 0.467 (definition 0.266), #9 translate 0.664 |
| 151 | 2 | 0.00 | 8/10 | #4 definition 0.387 (translate 0.363) ✗, #9 definition 0.461 ✗ |
| 150 + cutoff | 1 | 0.79 | 10/10 | none |
| 150 + cutoff | 2 | 0.52 | 10/10 | #9 translate 0.694 |

Notes: run on CPU in fp32 with the saved seed-0 weights; probabilities differ from the bf16 GPU test run by at most 0.002 and no prediction changes. Seeds 1-2 weights were not kept, so their rows come from `runs/m15*_s{1,2}/test_predictions.csv`.
