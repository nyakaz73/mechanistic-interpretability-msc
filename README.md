# Mechanistic interpretability MSc

Probing DistilBERT **MLP activations** for linearly decodable information about **certain** versus **uncertain** language in biomedical text.

**Repository:** [github.com/nyakaz73/mechanistic-interpretability-msc](https://github.com/nyakaz73/mechanistic-interpretability-msc)

---

## Repository layout

| Path | Role |
|---|---|
| `dataset.ipynb` | Build and document the BioScope certain / uncertain corpus |
| `dataset/bioscope_certainty_full.csv` | Full usable labelled table (after preprocessing) |
| `dataset/bioscope_certainty.csv` | Balanced probing table (equal class counts) |
| `mlp_confidence_probing.ipynb` | **Main experiment:** MLP hooks, layer sweep, probes, shuffled-label controls |
| `probing_transformers_playground.ipynb` | **Playground / tutorial** workbook (SST-2 sentiment probing practice) |
| `requirements.txt` | Python dependencies |

---

## Environment setup

```bash
cd mechanistic-interpretability-msc
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open notebooks from the **repository root** so relative paths such as `dataset/bioscope_certainty.csv` resolve correctly.

---

## Dataset documentation

### Source

The primary corpus is **[BioScope](https://rgai.inf.u-szeged.hu/node/105)** (Vincze et al., 2008), a public biomedical annotation set. Sentences come from:

| BioScope part | File | Used? |
|---|---|---|
| Abstract | `abstracts.xml` | Yes |
| Paper | `full_papers.xml` | Yes |
| Medical text | `clinical_records_anon.xml` | No (anonymised to `*`) |

The official zip is downloaded and parsed in `dataset.ipynb` (cached under `~/.cache/bioscope`). Hugging Face `bigbio/bioscope` is not used directly because modern `datasets` versions no longer run Hub loading scripts.

This is **not** a hand-written template set. Labels are derived from BioScope’s published speculation annotations.

### Generation / preprocessing process

Implemented in `dataset.ipynb` in this order:

1. **Import** all three BioScope sources into a raw table.
2. **Explore** document mix, medical `*` rows, speculation / negation flags.
3. **Preprocess**
   - drop all `Medical text` rows (tokens replaced by `*` in the release);
   - drop short alphabetic fragments (`MIN_ALPHA = 20` letters) such as headings (`Results`, `Methods`);
   - map speculation annotations to binary certain / uncertain labels;
   - undersample `certain` to match `uncertain` for the probing table (`random_state=42`);
   - write CSV files under `dataset/`.

### Number of observations

| Table | File | Rows |
|---|---|---|
| Full usable set | `dataset/bioscope_certainty_full.csv` | **14,433** |
| Balanced probing set | `dataset/bioscope_certainty.csv` | **5,240** |

Raw BioScope before cleaning contains 20,924 sentences (including 6,383 medical `*` rows).

### Class distribution

**Full usable set**

| Class | `label` | Count |
|---|---|---|
| certain | 1 | 11,813 |
| uncertain | 0 | 2,620 |

**By document type (full set)**

| document_type | certain | uncertain |
|---|---:|---:|
| Abstract | 9,758 | 2,101 |
| Paper | 2,055 | 519 |

**Balanced probing set**

| Class | Count |
|---|---:|
| certain | 2,620 |
| uncertain | 2,620 |

### Labelling procedure

BioScope does not ship a `certain` / `uncertain` column. It annotates **speculation** and **negation** cues. Binary labels are defined as:

| `label` | `label_name` | Rule |
|---|---|---|
| `1` | `certain` | no speculation cue in the sentence |
| `0` | `uncertain` | at least one speculation cue (e.g. *may*, *suggest*, *indicate that*) |

Negation is stored as `has_negation` and does **not** flip the class alone. A negated but non-speculative sentence (e.g. “no induction was detected”) remains `certain`.

Columns in the CSV files: `id`, `document_id`, `document_type`, `sentence_id`, `text`, `has_negation`, `label`, `label_name`, `speculation_cues`, `text_len`.

### Topic / document distribution

There are **no synthetic templates**. Topic structure follows BioScope’s biomedical sources:

- **Abstract** vs **Paper** (`document_type`);
- biomedical domains covered by the original BioScope collection (genetics, immunology, molecular biology, etc.).

Class balance differs by source in the full table (see crosstab above). The balanced CSV preserves both sources after undersampling.

### Examples

**Certain (`label = 1`)**

> Induction of NF-KB during monocyte differentiation by HIV type 1 infection.

**Uncertain (`label = 0`)**

> These results indicate that in monocytic cell lineage, HIV-1 could mimic some differentiation/activation stimuli allowing nuclear NF-KB expression.

### Train / validation / test splitting strategy

| Stage | Strategy |
|---|---|
| Dataset build (`dataset.ipynb`) | No train/val/test split. Produces full and balanced CSV tables only. |
| Main probing (`mlp_confidence_probing.ipynb`) | Stratified **train / test** split on the balanced table: `test_size=0.3`, `random_state=42` (`SEED=42`). The same indices are reused across all layers. |
| Validation set | **Not yet** a separate hold-out fold. Model selection currently uses the stratified test split plus a **shuffled-label control**. |
| Planned control (lexical leakage) | Future work: document-level and/or cue-held-out splits so speculation markers seen at test time are not all available at training time. |

---

## How to build the dataset

1. Install dependencies (see above).
2. Open `dataset.ipynb` from the repo root.
3. Run all cells top to bottom.

Expected outputs:

- `dataset/bioscope_certainty_full.csv`
- `dataset/bioscope_certainty.csv`

---

## How to run the main experiment (`mlp_confidence_probing.ipynb`)

This is the dissertation experiment notebook: DistilBERT forward hooks, MLP neuron extraction, layer-wise logistic probes, Pearson / Ridge neuron ranking, and shuffled-label controls.

1. Complete `dataset.ipynb` so `dataset/bioscope_certainty.csv` exists.
2. Open `mlp_confidence_probing.ipynb` from the **repo root**.
3. Run sections in order:
   - environment / model load (`SEED=42`, DistilBERT);
   - MLP activation extraction helpers;
   - qualitative confident vs uncertain comparison;
   - **Section 6:** load `dataset/bioscope_certainty.csv`;
   - **Section 7:** batch MLP features + layer sweep (`train_test_split`, `test_size=0.3`);
   - neuron ranking and validation controls.

Hardware: CPU is fine for DistilBERT; Apple Silicon MPS / CUDA speeds up activation extraction over 5,240 sentences.

---

## Playground / tutorial workbook

`probing_transformers_playground.ipynb` is a **practice / tutorial** notebook, not the thesis experiment.

It teaches probing concepts on **SST-2 sentiment** (GLUE): frozen DistilBERT hidden states, layer-wise linear probes, and shuffled-label controls. Use it to learn the probing workflow before (or alongside) the BioScope certainty experiment in `mlp_confidence_probing.ipynb`.

---

## Suggested reading order

1. `probing_transformers_playground.ipynb` — playground / tutorial  
2. `dataset.ipynb` — BioScope certain / uncertain corpus  
3. `mlp_confidence_probing.ipynb` — main MLP certainty probing experiment  

---

## Citation (dataset)

Vincze, V., Szarvas, G., Farkas, R., Móra, G. and Csirik, J. (2008) ‘The BioScope corpus: biomedical texts annotated for uncertainty, negation and their scopes’, *BMC Bioinformatics*, 9(Suppl 11), S9.  
Corpus page: https://rgai.inf.u-szeged.hu/node/105
