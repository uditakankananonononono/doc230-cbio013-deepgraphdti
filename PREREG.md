# PREREG - CBIO013 (2024) "Fighting Future Pandemics with Novel DeepGraphDTI" (Wan Qin Lim), CPU slice "013-G"

Status: written and committed BEFORE any KIBA/BindingDB/Davis file is downloaded or scored. Lock tag: `prereg-lock-v2` (v1 tag, local only and never pushed, was superseded before any data download to fix the D4/D5/P4 wording; no data had been touched).
Scope: this slice does NOT reimplement DeepGraphDTI and makes no claim about it. It tests the project's core idea in a CPU-feasible form: does a protein representation built from an atom-level molecular graph (the sequence converted to a molecular graph by RDKit, as the project describes) add predictive signal for drug-target affinity beyond sequence-only protein features, under cold-target splits? No GPU, no deep graph network, no docking.
Overlap, stated openly: doc230-cbio028x-ligand3d-davis tests 3D ligand descriptors on Davis (cold-drug); mega27-20-drug-target-prediction ran Davis benchmark audits. This slice differs: primary dataset KIBA (not used in either), cold-TARGET splits, protein-side representation is the variable.

## Data (fixed)
Primary: KIBA as distributed with DeepDTA (github hkmztrk/DeepDTA data/kiba): ligands_can.txt, proteins.txt, Y (KIBA score; NaN = missing, dropped). SHA256 recorded in data_manifest.json at download.
Secondary cross-check only: Davis (data/davis, pKd = -log10(Kd/1e9)); results reported but no gate depends on it except D7.
BindingDB: attempted only if a public bulk file downloads without login within the session; otherwise reported as "not reached". Not gated.

## Features (fixed)
Ligand (same for all directions): Morgan r2 1024-bit counts + RDKit 2D descriptor set (CalcMolDescriptors), NaN->0.
Protein variants:
 P0 amino-acid composition (20) + dipeptide composition (400).
 P1 hashed 3-mer counts (1024 bins).
 P2 atom-graph protein: RDKit MolFromSequence (heavy-atom graph of the full chain), Morgan radius 2 hashed counts (1024 bins) from the full-chain molecule, plus graph summaries (atoms, bonds, rings count, mean degree, Wiener-free summaries only: degree histogram 0-4).
 P3 = P0 + P2 concatenated.
 P4 control: per-ligand train-mean KIBA score (protein ignored; unseen ligands get the global train mean).
Proteins longer than 1000 residues are truncated to the first 1000 (as DeepDTA); counts of truncated proteins reported.
Model for every direction: LightGBM regressor, 300 trees, num_leaves 31, learning_rate 0.05, colsample_bytree 0.3, seed 0, n_jobs 2. No tuning.

## Splits (fixed)
Cold-target 5-fold: whole proteins held out; repeated with 3 fold assignments (seeds 0,1,2). Metrics pooled over test pairs per repetition: Pearson r, RMSE, and Concordance Index on a fixed random 200k-pair subsample (seed 0). Report mean and sd across repetitions.

## Directions and gates (each: pass/fail, descriptive, no post-hoc change)
D1 P0 + ligand, KIBA cold-target. Baseline; no gate.
D2 P1 + ligand, KIBA cold-target. Baseline; no gate.
D3 P2 + ligand, KIBA cold-target. Gate: r(D3) - max(r(D1), r(D2)) > 0.01.
D4 P3 + ligand, KIBA cold-target. Gate: r(D4) - r(D1) > 0.01.
D5 P4 control (per-ligand train mean). Gate: r(D1) - r(D5) > 0.05 (protein features carry signal beyond ligand identity). Report r(D5).
D6 Permutation control: shuffle the protein-feature rows among proteins (P2), cold-target, 3 seeds. Gate: r(D6) < r(D3) - 0.02 (the graph signal is protein-specific, not a ligand shortcut).
D7 Davis cross-check: D1 and D3 on Davis cold-target. Gate: sign of (r(D3)-r(D1)) on Davis equals sign on KIBA; magnitude reported.
D8 Scaffold-blind check: KIBA cold-target AND cold-ligand double split (hold out 20% proteins and 20% ligands per fold; train on the rest). Gate: r(D3) > r(D1) still holds (any positive margin), reported with 3 repetitions.
D9 Nipah/Pathogen-Box ranking (UNVALIDATED, no ground truth, no efficacy claim): rank compounds from the MMV Pathogen Box (if SMILES publicly downloadable; else a disclosed alternative public list) against NiV G (UniProt Q9IH62) and NiV F (UniProt Q9IH63) sequences with the D3 model trained on all KIBA. Reported only: top-20 lists, rank correlation between D1-model and D3-model rankings across the compounds. Gate: none. Predicted-affinity extrapolation to viral glycoproteins is out of domain (KIBA is kinase-heavy); stated as a limitation, not a finding.
D10 Applicability-domain audit: fraction of KIBA test proteins whose max 3-mer Jaccard similarity to training proteins exceeds 0.5, and r(D3) split by that fraction (high vs low). Gate: reported only.

## Rules
Run once. Verbatim results committed in RESULTS.md. A missed gate is reported as a miss. If D3 does not beat baselines, the honest negative stands and I pivot to the next direction rather than refitting. Any code change after the lock tag is recorded as an amendment with timestamp before rerun and the original numbers kept.
## Caveats
KIBA is kinase-dominated; cold-target splits on ~229 proteins give few independent targets, so fold variance is large. Atom-graph hashed counts are a lossy proxy for a learned GNN. No claim about DeepGraphDTI, Nipah inhibitors, or clinical relevance.
