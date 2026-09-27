# QuLit + HERMES

**Quantum-Linguistic Framework for Superposed Semantic Representation, extended with Reader-Conditioned Interpretation**

QuLit Research Group · 2025

| 98 | 6 | 35+ | 12 |
|---|---|---|---|
| Verses | Languages | Authors | Contexts / Emotional states |

This repository contains two connected research systems:

- **QuLit** — a quantum-inspired NLP framework that represents literary meaning as a *density matrix* instead of a single fixed vector, so that a verse's multiple valid interpretations can coexist mathematically until a "reader context" collapses them into one.
- **HERMES** (Human-Centered Emotional Reader-Observer Modeling System) — built directly on top of QuLit. Where QuLit models *the poem*, HERMES models *the poem and the specific person reading it* (their culture, era, and emotional state), and can generate a plain-English interpretation conditioned on that reader.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [QuLit: Core Framework](#2-qulit-core-framework)
3. [HERMES: From QuLit to a Reader Model](#3-hermes-from-qulit-to-a-reader-model)
4. [Dataset — QuLitBench v2](#4-dataset--qulitbench-v2)
5. [Results Summary](#5-results-summary)
6. [Repository Structure](#6-repository-structure)
7. [Notebook Execution Order](#7-notebook-execution-order)
8. [Dependencies & Setup](#8-dependencies--setup)
9. [Known Limitations](#9-known-limitations)
10. [Roadmap / Next Steps](#10-roadmap--next-steps)
11. [Novel Contributions](#11-novel-contributions)

---

## 1. Problem Statement

Literary, poetic, and culturally rich text is fundamentally ambiguous. A single verse — an Urdu couplet, a Sanskrit shloka, an Arabic qasida, a Tamil Sangam poem, an English metaphysical verse — rarely carries one meaning. It carries a *superposition* of meanings, and the "right" one only emerges in context: in the reader, the culture, the moment of reading.

Classical NLP models (BERT, GPT, XLM-R, LaBSE) represent meaning as a single fixed vector. A poem with three valid interpretations gets collapsed into one representation; the other meanings are simply lost, and two very different readers (say, a Sufi mystic and a feminist critic) get the identical output. QuLit and HERMES exist to fix this.

## 2. QuLit: Core Framework

### 2.1 The Core Idea

Quantum mechanics already has a rigorous mathematical framework for "multiple states coexisting until measured": **superposition**, captured by a **density matrix** (a matrix encoding a probability distribution over all possible states at once, rather than a single point).

| Classical NLP (mBERT) | QuLit (density matrix) |
|---|---|
| Single 768-dim vector — one fixed point | 16×16 matrix (256 numbers) encoding all meaning states |
| Output never changes with context | Off-diagonal terms encode coherence — meanings interfere |
| Cannot represent ambiguity | Diagonal = probability of each meaning state; trace = 1 |
| Two readers get identical output | Different "measurement operators" (contexts) collapse to different meanings |

### 2.2 Pipeline (Modules 1–5)

1. **Dataset Creation** — raw verse in any of 6 languages → QuLitBench JSON/CSV with 3 human-curated interpretations per verse.
2. **Classical Baseline** — mBERT (`bert-base-multilingual-cased`) embeds each verse into a 768-dim vector. This is the ceiling QuLit is built to beat.
3. **Quantum Encoder** — a 4-qubit, 2-layer Parameterized Quantum Circuit (PQC) turns the 768-dim embedding into a 16×16 density matrix via Hadamard superposition → RY angle-encoding → CNOT entanglement → learned RX/RY/RZ refinement → second entanglement.
4. **Quantum Measurement** — 12 context operators (grief, love, longing, joy, sufi, vedantic, feminist, political, literal, metaphorical, philosophical, ironic) are applied via `Expectation = Trace(ρ × M)` to collapse the density matrix into a context-specific meaning activation.
5. **Cross-Lingual Entanglement** — quantum fidelity and trace distance detect verses in *different* languages that share the same meaning space, without needing parallel corpora.

The PQC is trained (Phase 5) with contrastive metric learning (30 epochs, batch size 8, 80 positive/80 negative pairs) so that semantically similar verses are pulled together in quantum space and dissimilar ones pushed apart.

### 2.3 QLAS — Quantum Literary Ambiguity Score

QuLit's headline novel metric, combining three signals into one score in `[0, 1]`:

```
QLAS = 0.4·S + 0.3·C + 0.3·K

S = Spectral Entropy       (normalized Von Neumann entropy — many equally
                             likely meanings = deep ambiguity)
C = Coherence Magnitude    (average off-diagonal density-matrix magnitude —
                             meanings actively interfering = literary depth)
K = Contextual Sensitivity (std of activation across 8 diverse contexts —
                             meaning swings hard with reader context = rich)
```

### 2.4 IBM Quantum Hardware Validation

The circuit was ported from PennyLane (simulation) to Qiskit and run on real IBM quantum hardware (`SamplerV2`, 2048 shots) — the first literary-NLP study to do so. Finding: **noise effects scale with a verse's literary ambiguity** — higher-QLAS verses show larger (hardware entropy − simulator entropy) gaps than more direct verses. This is an empirical result with no known prior in the literature.

## 3. HERMES: From QuLit to a Reader Model

QuLit's "context" was a label — an on/off switch (`context = grief`) with no model of *who* the grieving reader actually is. HERMES replaces the switch with a full reader model, adding a 7-phase pipeline (H1–H7) on top of QuLit's corpus and quantum-geometry theory.

| # | What changed | QuLit | HERMES |
|---|---|---|---|
| 1 | Reader representation | Context as a tag | Reader modeled across 20 cultural traditions × 10 time periods × 12 emotional states → single 64-dim observer vector |
| 2 | Meaning | Fixed per verse | Function of verse **and** reader together |
| 3 | Positional encoding | Fixed formula | Learnable Fourier frequencies |
| 4 | Ambiguity | Property of the verse only | Property of verse **for a given reader** (Observer-Sensitive Index) |
| 5 | Output | Analysis / retrieval only | Adds a generation phase — writes a plain-English interpretation |
| 6 | Cross-verse comparison | Pairwise mBERT similarity | Full alignment graph across all 98 verses, 6 languages, using reader-conditioned vectors |
| 7 | Evaluation | Informal, qualitative | Formal 6-protocol suite + 7-configuration ablation study |

### 3.1 Phase Pipeline

```
QuLitBench v2 Corpus (98 verses · 6 languages · 35+ authors)
        ▼
H1 — Observer Encoder            Cultural × Temporal × Emotional → 64-dim vector
        ▼
H2 — Neural Semantic Field       Text + Observer → 128-dim meaning vector
        ▼
H3 — Quantum Geometry Module     Maps meaning into Hilbert space; entanglement / phase drift
        ▼
H4 — Observer-Sensitive Index    How ambiguous is this verse for THIS reader?
        ▼
H5 — Interpretation Generator    Soft-prefix T5 decoder writes the interpretation in words
        ▼
H6 — Cross-Cultural Aligner      Finds thematic twins across all 6 languages (r = 0.790)
        ▼
H7 — Master Evaluation Dashboard 6-protocol scorecard (E1–E6)
```

**H1 — Observer Encoder.** Cultural identity → 32-dim embedding (20 traditions), emotional state → 16-dim VAE bottleneck (12 emotions), temporal period → 8-dim embedding (10 periods); concatenated (56-dim) → fusion/residual layers → L2-normalized 64-dim observer vector. **99.49% context accuracy**, silhouette score **0.7445**.

**H2 — Neural Semantic Field.** mBERT text (768-dim) is compressed to 64-dim, expanded via learnable Fourier positional encoding to 576-dim, fused with the 64-dim observer embedding through 6 field layers (with a layer-3 skip connection), then split into a mean/log-variance head sampled via the VAE reparameterization trick to produce the 128-dim meaning vector. **74.49% interpretation accuracy** for the full system; removing the observer encoder drops this to 62.63% (a **+12-point** contribution from the reader model).

**H3 — Quantum Geometry Module.** Maps H2's meaning vectors into a Hilbert space, computes an entanglement metric between verse pairs and a "phase drift" as the observer changes, and models ambiguous verses as superpositions. Feeds H4 and H6; produces no standalone metric of its own.

**H4 — Observer-Sensitive Index (OSI).** `OSI(v, O) = √[ Σ P(mᵢ|O)·(mᵢ − μ_O)² ]` — the standard deviation of the meaning distribution for a given reader. Observed range **0.12–0.89**; highest for mystical poets (Al-Hallaj, Kabir). Correlation with human ambiguity ratings is currently weak (**Pearson r = 0.14, p = 0.15**, not significant).

**H5 — Interpretation Generator.** Meaning vector → 10 soft-prefix tokens (5,120 values) for a T5-small decoder; observer vector adds a bias; OSI gates how exploratory vs. direct the generation is. **Weakest phase**: the T5-small decoder collapsed under only 98 training verses, repeating the same phrase ~7/10 times in the survey run. Meaning vectors are computed correctly — the decoder is undersized.

**H6 — Cross-Cultural Aligner.** Builds a full alignment graph over all 98 verses using reader-conditioned meaning vectors. **Pearson r = 0.790**, beating mBERT cosine (0.753) and Jaccard similarity (0.744). Notable discoveries: Ghalib ↔ Al-Hallaj (0.840, mystical longing), Kabir ↔ Upanishad (0.812, non-dual consciousness), and Tamil verses (Kaniyan Pungundranar, Thiruvalluvar) emerging as the most cross-culturally central nodes in the graph.

**H7 — Master Evaluation Dashboard.** Aggregates six evaluation protocols (E1–E6), summarized below.

## 4. Dataset — QuLitBench v2

The first annotated multilingual literary ambiguity corpus, hand-curated (not machine-generated).

| Property | v1 | v2 |
|---|---|---|
| Total verses | 8 | **98** |
| Languages | 6 | 6 (Urdu, Hindi, Sanskrit, Arabic, English, Tamil) |
| Interpretations / verse | 3 | 3 (all independently valid, not ranked) |
| Total interpretations | 24 | 294 |
| Ambiguity type labels | 4 | 4 (metaphorical, emotional, cultural, philosophical) |
| Unique authors | 7 | 35+ |
| Time span | 200 BCE – 1950 CE | 200 BCE – 1950 CE |

Language coverage (v2): Urdu 17 verses (Ghalib, Mir Taqi Mir, Faiz), Hindi 16 (Kabir, Mahadevi Varma, Dinkar, Bachchan), English 17 (Shakespeare, Donne, Keats, Frost, Eliot, Whitman, Dickinson, Yeats, Thomas), Sanskrit 16 (Upanishads, Gita, Kalidasa, Shankaracharya), Tamil 16 (Thiruvalluvar, Tirumular, Kaniyan Pungundranar), Arabic 16 (Al-Mutanabbi, Al-Hallaj, Ahmad Shawqi, Imru al-Qays).

Ambiguity type distribution: philosophical ~45%, emotional ~25%, cultural ~20%, metaphorical ~10%.

Each entry follows a strict 9-field schema: `id, language, script, romanized, translation, interpretations[3], ambiguity_type[], source, author`.

## 5. Results Summary

### 5.1 QuLit — vs. classical baselines (mBERT, XLM-R, LaBSE, SBERT)

| Task | Metric | Best classical baseline | QuLit |
|---|---|---|---|
| Task 1 — Ambiguity Resolution (ISS) | Mean ISS ↑ | 0.0210 (LaBSE) | **0.0580** |
| Task 2 — Cross-Lingual Similarity | Pearson r ↑ | 0.5823 (LaBSE) | **0.6247** |
| Task 3 — Genre Classification | F1 ↑ | 0.4287 (LaBSE) | **0.5143** |

Feature-importance analysis shows the 12 context-activation features dominate the genre classifier, supporting the core claim that context-sensitive quantum measurement captures real literary structure.

### 5.2 HERMES — What worked, what didn't

| Status | Phase / Protocol | Result |
|---|---|---|
| ✅ Strong | H1 Observer Encoder | 99.49% context accuracy, 0.7445 silhouette |
| ✅ Strong | H6 Cross-Cultural Alignment | r = 0.790, beats mBERT (0.753) |
| ✅ Strong | H2 + reader profile (ablation) | +12 pts accuracy (74.49% vs 62.63%) |
| ✅ Complete | E5 Ablation Study | 7 configs — every component contributes |
| ⚠️ Partial | H5 Generation | Meaning vectors correct; T5-small decoder collapses/repeats |
| ⚠️ Partial | E4 Cross-Lingual Retrieval | H@1 = 0.20 vs mBERT 0.60, but ties at H@3 (0.60) and H@5 (0.80) |
| ❌ Needs work | E3 Ambiguity Calibration (OSI) | r = 0.14, p = 0.15 — not significant yet |
| ❌ Needs work | E1 Direction Accuracy | 36.16% (barely above 33.3% random); anomaly — removing the observer *improves* this metric |
| ⏳ Pending | E2 Human Evaluation | Survey designed (14 verse pairs); needs 3+ real annotators (currently 1: Agnes) |

## 6. Repository Structure

```
QuLit/
├── corpus/
│   ├── QuLitBench_v1.json / .csv        (8 verses)
│   └── QuLitBench_v2.json / .csv        (98 verses)
├── models/
│   ├── classical_embeddings(_v2).pkl
│   ├── quantum_embeddings.pkl
│   ├── trained_quantum_embeddings.pkl
│   ├── trained_pqc_params.pkl
│   ├── measurement_results.pkl
│   ├── entanglement_results.pkl
│   ├── baseline_embeddings.pkl
│   ├── all_baseline_results.pkl
│   └── ibm_quantum_results.pkl
├── results/
│   ├── classical_vs_quantum.png
│   ├── meaning_collapse_all.png
│   ├── one_verse_six_contexts.png
│   ├── semantic_entropy_chart.png
│   ├── training_summary_clean.png
│   ├── entanglement_matrices.png / entanglement_network.png
│   ├── simulator_vs_hardware.png / quantum_noise_analysis.png
│   ├── all_baselines_comparison.png
│   ├── qlas_analysis.png / ablation_study.png
│   ├── explorer_UR002.png / explorer_TA001.png / explorer_EN007.png / explorer_comparison.png
│   └── paper_results_table.csv
└── paper/
    └── human_eval_form.txt
```

*(HERMES phases H1–H7 build on this same corpus/model layout — see notebook list below.)*

## 7. Notebook Execution Order

**QuLit (Phases 1–12):**
```
1.  QuLit_Setup.ipynb                # install libraries
2.  QuLit_Dataset.ipynb              # create v1 dataset
3.  QuLit_Dataset_Expansion.ipynb    # expand to v2 (98 verses)
4.  QuLit_Classical_Baseline.ipynb   # mBERT embeddings
5.  QuLit_Quantum_Encoder.ipynb      # density matrices
6.  QuLit_Measurement.ipynb          # context operators + entropy
7.  QuLit_PQC_Training.ipynb         # train PQC (needs v2 embeddings)
8.  QuLit_Entanglement.ipynb         # cross-lingual analysis
9.  QuLit_Evaluation.ipynb           # initial evaluation
10. QuLit_IBM_Quantum.ipynb          # hardware execution
11. QuLit_Baselines.ipynb            # 5-model comparison
12. QuLit_Phase5.ipynb               # QLAS + ablation + human eval
13. QuLit_Interactive_Explorer.ipynb # demo system
```

**HERMES (Phases H1–H7)** run after the QuLit corpus and quantum-geometry outputs exist, in phase order (H1 → H7) as laid out in Section 3.1 above.

## 8. Dependencies & Setup

| Library | Version | Purpose |
|---|---|---|
| pennylane | 0.44.1 | Quantum circuit simulation & PQC training |
| qiskit | latest | IBM Quantum hardware interface |
| qiskit-ibm-runtime | latest | `SamplerV2` for real hardware execution |
| transformers | 5.0.0 | mBERT / XLM-R embeddings |
| sentence-transformers | latest | LaBSE / SBERT embeddings |
| torch | 2.10.0 | PyTorch backend |
| scikit-learn | latest | KNN, SVM, PCA, evaluation metrics |
| scipy | latest | Pearson / Spearman correlation |
| ipywidgets | latest | Interactive Explorer UI |
| matplotlib | latest | Visualizations |

HERMES additionally requires a T5 checkpoint (T5-small) for H5, and note that its checkpoint is saved across **three separate keys** — `proj_state`, `t5_dec_state`, `t5_lm_state` — so it cannot be reloaded with a plain `load_state_dict` call.

To run on real quantum hardware (QuLit Phase 7), you'll need an IBM Quantum account/token (`QiskitRuntimeService`).

## 9. Known Limitations

Honest, in priority order:

1. **H5 decoder collapse** — T5-small is too small for only 98 training verses; it repeats safe phrases. *Fix:* condition a larger LM (Gemini/Claude/GPT) on the meaning + observer vectors instead of fine-tuning T5 further.
2. **E3 OSI not yet significant** — r = 0.14, p = 0.15 against human ambiguity ratings. *Fix:* more GPU training on H2 for more discriminative meaning vectors (OSI is derived from that distribution).
3. **E2 relies on synthetic ratings** — currently a RoBERTa-based proxy, not acceptable for ACL/EMNLP-level publication. *Fix:* 3+ real annotators rating the 14 already-designed verse pairs.
4. **E1 direction-accuracy anomaly** — removing the observer encoder *increases* direction accuracy (36.16% → 41.25%), which is counterintuitive. *Fix:* audit the soft-target/context-to-interpretation mapping in `NSFDataset`.
5. **Single annotator (E2 pilot)** — only one rater (Agnes) so far, so inter-annotator agreement (Cohen's κ) can't be computed. *Fix:* recruit 2+ more annotators; κ > 0.4 is publishable for subjective literary tasks.
6. **Citation verification pending** — 5 citations flagged as potentially hallucinated/unverifiable by a Q-Data review; need individual checking against published sources.
7. **Small corpus** — 98 verses is small for deep learning; root cause of several of the above issues. *Fix:* expand QuLitBench (even doubling to ~200 verses would meaningfully help training stability).

## 10. Roadmap / Next Steps

In priority order:

1. Recruit 2 more human annotators to rate the 14 verse pairs (single most important remaining task).
2. Run H2 for more epochs on GPU to improve E3 (OSI calibration) and E4 (retrieval ranking).
3. Regenerate the 6 duplicate verse pairs in E2 via the Gemini API pipeline.
4. Investigate and fix the E1 direction-accuracy anomaly in H2's soft-target mapping.
5. Verify the 5 flagged citations from the Q-Data review.
6. Upgrade the H5 generation backend to a larger LM conditioned on the meaning/observer vectors.
7. Expand QuLitBench v2 with additional verses per tradition.

## 11. Novel Contributions

- **QuLitBench** — first annotated multilingual literary ambiguity corpus (3 interpretations/verse, 6 languages, 2,200+ years of tradition).
- **Quantum Semantic Encoder** — first use of a PQC to convert literary text into density matrices via Hadamard superposition + CNOT entanglement.
- **Context-Sensitive Measurement** — first application of quantum measurement theory to reader-context-dependent literary meaning collapse.
- **Cross-Lingual Quantum Entanglement** — detects cross-lingual meaning kinship via quantum fidelity / trace distance, without parallel corpora.
- **QLAS** — first quantum-theoretic literary ambiguity metric (spectral entropy + coherence + contextual sensitivity).
- **IBM Quantum Hardware Study** — first literary-NLP paper run on real quantum hardware; found that hardware noise correlates with semantic complexity.
- **Interactive Meaning Explorer** — real-time system accepting any verse, in any language, and visualizing meaning collapse across contexts.
- **HERMES Reader Model (H1–H7)** — first system to condition literary-meaning representation jointly on the text *and* an explicit, learned model of the reader (culture × era × emotion), including a reader-specific ambiguity index (OSI) and a reader-conditioned interpretation generator.

---

*This README summarizes the QuLit Technical Documentation and the HERMES Detailed Document. Refer to those source documents for full mathematical derivations, complete code listings, and per-notebook implementation detail.*
