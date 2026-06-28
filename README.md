# TSItransRL

TSItransRL is a Python codebase for **goal-directed molecular generation** with a staged workflow:

1. **Teacher model training / update** (`1_train.py`)
2. **Distillation to a Hugging Face GPT-2 student** (`2_distill.py`)
3. **Molecule generation + evaluation** (`3_generate.py`, `eval_script.py`, `1_1_gen.py`)

The repository is currently configured mainly for the **JNK3 + GSK3 multi-objective task** (with QED/SA constraints), while parts of the code still include commented alternatives for DRD2.

---

## Repository Structure

- `1_train.py` — trains/continues a custom conditional Transformer decoder (`MoleGPT` in `model.py`).
- `1_1_gen.py` — generates molecules from `MoleGPT` checkpoints and computes MOSES/property metrics.
- `2_distill.py` — fine-tunes/distills a GPT-2 LM (`GPT2LMHeadModel`) on selected generated molecules.
- `3_generate.py` — samples from the distilled GPT model and evaluates outputs.
- `eval_script.py` — shared evaluation logic (`valid_metric`) for validity/novelty/diversity/success metrics.
- `model.py` — custom autoregressive decoder implementation + scheduler.
- `data.py` — dataset and CSV parsing utilities.
- `utils.py` — property predictors (DRD2/JNK3/GSK3), SA/QED helpers, utility scoring logic.
- `properties.py` — additional property scoring helpers (older/parallel utility module).

---

## What the pipeline does

At a high level, the code uses molecular SMILES generation conditioned by task labels and iteratively improves generation quality:

- Label molecules as `positive`/`negative` using task constraints.
- Train/update a conditional generator (`MoleGPT`).
- Generate candidate molecules.
- Filter/select candidates according to activity + drug-likeness constraints.
- Distill selected molecules into a GPT-2 style student model.
- Sample and evaluate final generated molecules with validity/novelty/diversity and success metrics.

Current success criterion in active code path (JNK3/GSK3):

- `jnk3 >= 0.5`
- `gsk3 >= 0.5`
- `qed >= 0.6`
- `sa <= 4.0`

---

## Environment

> The project requires Python + PyTorch + RDKit + Hugging Face Transformers + MOSES.  
> CUDA is assumed by many scripts (`.cuda()` is used directly).

Recommended:

- Python 3.9/3.10
- CUDA-enabled PyTorch
- RDKit (conda installation recommended)

Example setup (conda):

```bash
conda create -n tsitransrl python=3.10 -y
conda activate tsitransrl

# Install PyTorch matching your CUDA version (example command may vary):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install transformers pandas numpy tqdm seaborn matplotlib scikit-learn

# RDKit (recommended via conda-forge)
conda install -c conda-forge rdkit -y

# MOSES metrics package
pip install molsets
```

If `molsets` does not expose the `moses` namespace in your environment, install a compatible MOSES package used by your platform.

---

## Data and assets expected by code

The scripts assume multiple local assets already exist. Important examples from hardcoded paths:

- Tokenizer/model directories:
  - `./liuyingv3`
  - `./jnk_gsk_res/liuyingdistill*`
  - `./jnk_gsk_res/YingGPTModel`
- Activity/classifier files:
  - `./drd2data/drd2.pkl`
  - `./drd2data/jnk_gsk/jnk3.pkl`
  - `./drd2data/jnk_gsk/gsk3.pkl`
- Training/eval CSVs:
  - `./jnk_gsk_res/jnk_gsk_train_new7.csv`
  - `./jnk_gsk_res/jnk_gsk_test_new7.csv`
  - `./drd2data/jnk_gsk/jnk_gsk_activity.csv`

CSV schema expected in training data (see `data.py`):

- Must contain `SMILES`
- For JNK/GSK labeling path: columns `jnk3`, `gsk3`, `qed`, `sa`

---

## Stage-by-stage usage

### 1) Train / update teacher (`MoleGPT`)

```bash
python 1_train.py
```

Notes:

- Active paths/hyperparameters are set in `if __name__ == "__main__":`.
- The script currently **loads an existing checkpoint before training**:
  - `./jnk_gsk_res/molegpt_jnk_gsk6.ckpt`
- Output checkpoint is currently configured as:
  - `./jnk_gsk_res/molegpt_jnk_gsk7.ckpt`

### 2) Generate from teacher and evaluate

```bash
python 1_1_gen.py
```

This script:

- Loads `MoleGPT` checkpoint
- Samples molecules using conditional token `"<s>positive[SEP]"`
- Computes MOSES + property statistics
- Writes generated molecules/results to configured CSV/TXT paths

### 3) Distill into GPT-2 student

```bash
python 2_distill.py
```

This script:

- Loads a previous distilled model/tokenizer directory (currently `liuyingdistill6`)
- Reads selected molecules from CSV (currently `for_gpt_7.csv`)
- Freezes most Transformer layers and unfreezes the last blocks
- Trains with Hugging Face `Trainer`
- Saves to configured output dir (currently `./jnk_gsk_res/liuyingdistill7/`)

### 4) Sample from distilled model and evaluate

```bash
python 3_generate.py
```

This script:

- Loads a GPT model/tokenizer directory (currently `./jnk_gsk_res/YingGPTModel`)
- Samples molecules via `model.generate(...)`
- Saves generated SMILES CSV
- Calls `valid_metric(...)` from `eval_script.py`

---

## Evaluation metrics

The evaluation scripts compute (depending on call path):

- **Validity** (`fraction_valid`)
- **Novelty** (vs reference active set)
- **Uniqueness**
- **MOSES-style distribution metrics**:
  - SNN
  - Frag
  - Scaf
  - FCD
  - Internal diversity (`IntDiv`, `IntDiv2`)
  - Filter pass rate
- **Property stats/similarity** for `logP`, `SA`, `QED`, `weight`
- **Task success / real success** under threshold constraints

Outputs are written to CSV/TXT paths configured in each script.

---

## Important caveats before running

1. **Hardcoded paths**
   - Most scripts are experiment-style and use hardcoded local files/directories.
   - Update paths in each `__main__` block before running.

2. **GPU assumptions**
   - `.cuda()` is called directly in several places.
   - For CPU-only usage, code edits are required.

3. **Potential bug in `eval_script.py` main call**
   - `valid_metric` signature is:
     - `valid_metric(gen_csv, n_jobs, ref_path, gen_smiles_csv, res_path)`
   - But current `eval_script.py` main block calls:
     - `valid_metric(n_jobs, ref_path, gen_smiles_csv, res_path)`
   - This is missing the first argument (`gen_csv`) and has shifted positions.

4. **External/local dependencies not included in repo**
   - Tokenizer/model directories and `.pkl` predictors are required but not stored here.

5. **`pytorchtools.EarlyStopping`**
   - `1_train.py` imports `EarlyStopping` from `pytorchtools`, but that file/module is not present in this repository snapshot.

---

## Minimal quick start checklist

1. Prepare all required local assets (tokenizer dirs, checkpoints, `.pkl` predictors, CSVs).
2. Verify path settings in:
   - `1_train.py`
   - `1_1_gen.py`
   - `2_distill.py`
   - `3_generate.py`
3. Ensure `pytorchtools.py` is available in Python path (or replace with your own early stopping implementation).
4. Run stages in order:

```bash
python 1_train.py
python 1_1_gen.py
python 2_distill.py
python 3_generate.py
```

---

## Citation / attribution

This repository is a fork of:

- https://github.com/liuyingying95/TSItransRL

If you use this code in research, please cite the corresponding original work/repository as appropriate.
