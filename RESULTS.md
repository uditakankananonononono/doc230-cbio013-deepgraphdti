# RESULTS - CBIO013-G (run once; protocol PREREG.md @ prereg-lock-v2 + AMENDMENTS.md)

Numbers are verbatim from results/kiba_results.json, results/davis_results.json, results/gates.json. Mean ± sd over 3 cold-target fold assignments; every model trained on <=20,000 sampled training pairs (Amendment 3); all held-out pairs scored.

## KIBA cold-target (118,254 pairs, 2,111 ligands, 229 proteins)

| Dir | Setup | r | RMSE | CI |
|---|---|---|---|---|
| D1 | P0 AAC+dipeptide + ligand | 0.6022 ± 0.0047 | 0.6726 | 0.6970 |
| D2 | P1 hashed 3-mer + ligand | 0.6044 ± 0.0028 | 0.6709 | 0.7039 |
| D3 | P2 atom-graph protein + ligand | 0.6039 ± 0.0035 | 0.6682 | 0.6989 |
| D4 | P3 (P0+P2) + ligand | 0.6076 ± 0.0048 | 0.6681 | 0.7049 |
| D5 | per-ligand train mean (protein ignored) | 0.6266 ± 0.0010 | 0.6526 | 0.7058 |
| D6 | P2 with protein rows permuted | 0.5322 ± 0.0026 | 0.7086 | 0.6647 |
| D8_P0 | P0, cold-target AND cold-ligand | 0.4884 ± 0.0063 | 0.7276 | 0.6571 |
| D8_P2 | P2, cold-target AND cold-ligand | 0.4896 ± 0.0091 | 0.7268 | 0.6551 |

## Davis cross-check (cold-target)

| Dir | r | RMSE |
|---|---|---|
| D1 | 0.7003 ± 0.0006 | 0.6401 |
| D3 | 0.6890 ± 0.0023 | 0.6490 |

## D10 applicability domain (seed-0 folds, D3 predictions)
Fraction of test proteins with max 3-mer Jaccard > 0.5 to training proteins: 0.0349. r on pairs of high-similarity proteins 0.4191 (n=3835); low-similarity 0.6120 (n=114419).

## Gates (as preregistered)

- D3: NOT MET
- D4: NOT MET
- D5: NOT MET
- D6: MET
- D7: NOT MET
- D8: MET

## Reading (descriptive, no claim beyond data)
- D3 NOT MET: atom-graph protein features (r 0.6039) did not beat the best baseline (D2 3-mer, r 0.6044) by +0.01. D4 NOT MET (+0.0054 over D1, bar +0.01). The graph representation adds no measurable signal over sequence-only features at this scale.
- D5 NOT MET, and informative: a per-ligand train mean with no protein information (r 0.6266) beat every protein-feature model. Under cold-target splits on KIBA, ligand identity explains more than these protein features do. This is a limit of the benchmark and of these hashed features, not evidence about DeepGraphDTI.
- D6 MET: permuting protein rows drops r to 0.5322, so the protein features carry some protein-specific signal, but D1-D4 sit within 0.006 of each other.
- D7 NOT MET: on Davis the graph variant is lower (0.6890 vs 0.7003); sign differs from KIBA's tiny positive D3-D1 gap.
- D8 MET by the preregistered rule (any positive margin: 0.4896 vs 0.4884) but the gap 0.0012 is inside fold sd (0.006-0.009); not a finding.
- D9 (Nipah/Pathogen Box ranking) NOT RUN. Given D3/D5, a model with no protein advantage cannot support a target-specific ranking; it is dropped as uninformative, not as an efficacy claim.

## Limitations
20,000-pair training subsample (memory limit of 2 GB box); hashed/lossy graph features, not a learned GNN; KIBA is kinase-dominated; 229 proteins; no claim about DeepGraphDTI, Nipah, or any drug.
