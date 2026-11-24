# Race encoding and fairness of electrocardiogram foundation models across two cohorts

Code for the study of the same title: an audit of three frozen electrocardiogram (ECG) foundation models (ECGFounder, ECG-FM and HuBERT-ECG) in two hospital cohorts with echocardiography-confirmed labels (EchoNext and a replication cohort from MIMIC-IV). The notebooks go from the raw PhysioNet data to every table and figure of the paper and its Supplementary Information.

> Authors, citation and licence: to be added on publication.

## What the study does

Foundation models for the ECG are reused as frozen encoders for many detection tasks, so whatever their embeddings carry is available to every detector built on them. The study asks four questions, each answered with linear models on one frozen embedding per ECG and patient-level five-fold cross-fitting (every patient is scored once, by models that never saw any of that patient's ECGs):

| Question | How it is answered | Notebook |
|---|---|---|
| 1. Can race and ethnicity be decoded from the embeddings? | linear probes (Black versus White, and four groups), compared with the same architectures with random weights and with ECG measurements, age and sex | 05 |
| 2. Is this explained by cardiac structure? | a model on structural variables alone; probes among patients without recorded structural heart disease (SHD); matched Black–White pairs; probes after removing structural components from the embeddings | 06 |
| 3. Do detectors built on the embeddings make different errors between groups? | one detector per diagnosis with one threshold for every group; differences from White patients in flagging rate (a shifted operating point) and balanced accuracy (separation), standardised for age and sex | 07, 09 |
| 4. What does erasing race information do to these errors? | least-squares concept erasure (LEACE), fitted on all fitting ECGs or on ECGs without the diagnosis, and the detectors refitted | 08 |

## Contents

Everything is in the notebooks, to be run in this order. There are no Python modules: settings and shared functions live in `00_settings.ipynb`, which every other notebook runs with `%run ./00_settings.ipynb`.

| Notebook | Content | Paper items | Needs |
|---|---|---|---|
| `00_settings.ipynb` | **global settings** (the master folder and all paths), fixed constants, shared functions | – | – |
| `01_environment_setup.ipynb` | how to build the environment; checks of the packages and the GPU | – | – |
| `02_data_echonext.ipynb` | EchoNext: download instructions, cohort, five-fold design, characteristics | Table 1, Supplementary Figs. 1–2 | EchoNext |
| `03_data_mimic.ipynb` | MIMIC-IV: download instructions, cohort and labels, waveforms, five-fold design, characteristics, constants for the conversion to the EchoNext format | Table 1, Supplementary Fig. 1, Supplementary Table 2 | MIMIC-IV modules |
| `04_foundation_model_embeddings.ipynb` | downloads the three models (pinned revisions, SHA-256 checked) and extracts one frozen embedding per ECG | Supplementary Table 1 | 02, 03, GPU |
| `05_race_probes.ipynb` | question 1 | Fig. 1b, Supplementary Table 3, Table 2 | 04 |
| `06_cardiac_structure.ipynb` | question 2 | Fig. 1c, Supplementary Table 4, Table 2 | 05 |
| `07_disease_detectors.ipynb` | question 3: detectors for every diagnosis, errors by group | Fig. 2, Supplementary Tables 5–6, Table 2 | 04 |
| `08_race_erasure.ipynb` | question 4: standard and disease-negative erasure | Figs. 3–4, Supplementary Tables 3 and 7, Table 2 | 07 |
| `09_prevalence_equalised_refit.ipynb` | composite SHD detectors refitted after equalising prevalence across groups in the fitting data | Supplementary Fig. 3, Table 2 | 07 |
| `10_main_tables_and_figures.ipynb` | Table 1, Figs. 1–4, Table 2 | main text | 02–09 |
| `11_supplementary_tables_and_figures.ipynb` | Supplementary Figs. 1–3, Supplementary Tables 1–7 | Supplementary Information | 02–09 |

In notebooks 10 and 11 every table and figure has its own heading named as in the paper (`## Table 2 | …`, `## Supplementary Fig. 3 | …`), with its legend in the markdown cell above the code.

## Getting started

1. **Environment** (Python 3.12, built with [uv](https://docs.astral.sh/uv/); the versions used for the paper are pinned in `requirements.txt`). In the folder of this repository:

   ```bash
   uv venv --python 3.12 .venv
   source .venv/bin/activate                    # Windows PowerShell: .venv\Scripts\Activate.ps1
   uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # CPU only: https://download.pytorch.org/whl/cpu
   uv pip install -r requirements.txt
   python -m ipykernel install --user --name ecgfm-race --display-name "ECG-FM race encoding (Python 3.12)"
   jupyter lab
   ```

   Choose the kernel *ECG-FM race encoding (Python 3.12)* for every notebook. `01_environment_setup.ipynb` checks the installed versions and the GPU.

2. **Master folder.** Choose a folder with enough free space (see *Data* below) and either set the environment variable `ECGFM_RACE_DIR` to it, or edit the line `MASTER_DIR = ...` in the first code cell of `00_settings.ipynb`. Below it, `data/` holds the raw downloads, `models/` the pretrained weights, `work/` the row-level intermediate files, `results/` the aggregate results and `outputs/` the tables and figures. If your data are elsewhere, set `ECHONEXT_DIR`, `MIMIC_IV_DIR`, `MIMIC_ED_DIR`, `MIMIC_ECG_DIR` and `MIMIC_ECHO_DIR` (or edit the same cell).

3. **Data and pretrained models** are not part of this repository. Notebooks 02 and 03 explain for each dataset where it comes from, what access it requires and where the files go; notebook 04 downloads the pretrained weights.

4. **Run the notebooks in order.** Long steps skip what already exists, so they can be interrupted and resumed. A notebook can also be run without opening it, for example:

   ```bash
   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output executed/05_race_probes.ipynb 05_race_probes.ipynb
   ```

## Data

| Dataset (version) | Used for | Page | Access |
|---|---|---|---|
| EchoNext (1.1.1) | primary cohort: ECG waveforms, echocardiographic labels, race and ethnicity | https://physionet.org/content/echonext/1.1.1/ | restricted: PhysioNet account and the project's data use agreement |
| MIMIC-IV (3.1), `hosp/admissions`, `hosp/patients` | replication cohort: age, sex, race and ethnicity, admissions | https://physionet.org/content/mimiciv/3.1/ | credentialed: credentialed PhysioNet account, required training and the data use agreement |
| MIMIC-IV-ED (2.2), `ed/edstays` | race and ethnicity, emergency department stays | https://physionet.org/content/mimic-iv-ed/2.2/ | credentialed |
| MIMIC-IV-Echo (1.0.1), `structured-measurement.csv.gz` | echocardiographic labels and structural variables (the images are not needed) | https://physionet.org/content/mimic-iv-echo/1.0.1/ | credentialed |
| MIMIC-IV-ECG (1.0) | ECG waveforms and machine measurements | https://physionet.org/content/mimic-iv-ecg/1.0/ | open access |

The exact files and download commands are in notebooks 02 and 03; read the terms of each dataset on its own page before you use it.
Disk space: EchoNext 24 GB; MIMIC-IV-ECG 90 GB if downloaded in full, or about 27 GB for only the records used (notebook 03 writes their list); the MIMIC-IV waveform array 26 GB; embeddings 7 GB; model weights 2 GB.

**Data use.** The PhysioNet data use agreements do not allow individual-level data to be shared, and this includes derived embeddings and predictions. The notebooks print and save only aggregate values (counts, AUROCs, differences between groups) under `results/` and `outputs/`; row-level files stay in `work/`. Never commit `data/`, `models/`, `work/` or executed notebooks; `.gitignore` excludes them.

## Pretrained models

| Model | Source (pinned revision) | Licence stated by its authors |
|---|---|---|
| ECGFounder | https://huggingface.co/PKUDigitalHealth/ECGFounder (`d9b17939`); the architecture file `net1d.py` from https://github.com/PKUDigitalHealth/ECGFounder (branch `main`, `68d25f25`) | MIT (model card and repository) |
| ECG-FM | https://huggingface.co/wanglab/ecg-fm (`584219ea`), `mimic_iv_ecg_physionet_pretrained.pt` | MIT |
| HuBERT-ECG, large version | https://huggingface.co/Edoardo-Coppola/hubert-ecg-large (`f297337d`) | CC BY-NC 4.0 (non-commercial use only) |

No weights or code of these projects are included in this repository: notebook 04 downloads them and checks their SHA-256 hashes. The ECG-FM network is re-implemented in plain PyTorch following fairseq-signals, so that the published state dictionary loads strictly. Respect the licence of each model when you use its weights.

## Notes

- **Reproducibility.** The notebooks use the folds, seeds, solvers, grids of the regularisation strength and library versions of the analysis. On other hardware or other library versions, the last digit of some numbers can differ, because floating-point arithmetic on graphics cards and in multithreaded libraries is not identical across machines.
- **Hardware used:** Windows 11, NVIDIA GeForce RTX 5090 (32 GB; the notebooks cap their use at 28% of the GPU memory, `GPU_MEMORY_FRACTION` in notebook 00), 64 GB RAM. Notebook 04 needs a GPU; notebooks 05–09 run their model fits on the GPU and fall back to the CPU (slowly).
- **Running times** on this computer are given at the top of each notebook and in `01_environment_setup.ipynb`.
- The notebooks are stored without outputs.

## Repository layout

```
├── README.md
├── requirements.txt       package versions used for the paper (PyTorch is installed separately)
├── .gitignore
└── 00_settings.ipynb … 11_supplementary_tables_and_figures.ipynb
```
