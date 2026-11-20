| system | config | in_acc | oos_recall | oos_precision | oos_f1 | in_macro_f1 | auroc | overall_acc |
|---|---|---|---|---|---|---|---|---|
| TF-IDF 151 | C=10 | 92.1 [91.4, 92.9] | 55.5 [52.1, 58.4] | 89.8 [87.4, 92.1] | 68.6 [65.8, 71.0] | 88.6 [87.7, 89.3] | 95.3 [94.7, 96.0] | 85.5 [84.5, 86.4] |
| TF-IDF 150 | C=30, t=0.15 | 92.4 [91.6, 93.1] | 47.7 [44.5, 51.0] | 87.4 [84.5, 90.1] | 61.7 [58.8, 64.7] | 88.3 [87.3, 88.9] | 95.0 [94.3, 95.7] | 84.2 [83.3, 85.1] |

151: refuse iff argmax = oos; AUROC on P(oos). 150: refuse iff MSP < t, t in {0, 0.01, ..., 0.99} chosen on val by OOS F1; AUROC on 1 - MSP. C chosen on val by 151-way accuracy (151) or in-scope accuracy (150). Deterministic model: [95% bootstrap CI over test queries].
