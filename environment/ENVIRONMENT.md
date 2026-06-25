# Training and Inference Environment

This package records the cloud GPU environment used for the final SEDR-m6A/rank1 runs, not the local Windows environment.

## Python

- Python: `3.12.3`
- Python executable during training: `/root/miniconda3/bin/python`

## Hardware

- GPU class observed in the training logs: NVIDIA GeForce RTX 4080 SUPER
- CUDA wheel family: CUDA 12.4 (`torch==2.5.1+cu124`)

## Core Packages

```text
python==3.12.3
torch==2.5.1+cu124
transformers==5.8.1
multimolecule==0.1.0
numpy==2.1.3
pandas==3.0.3
scikit-learn==1.8.0
scipy==1.17.1
safetensors==0.7.0
tokenizers==0.22.2
tqdm==4.67.1
```

Install PyTorch from the CUDA 12.4 wheel index, then install the remaining packages from `requirements.txt`.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1+cu124
python -m pip install -r environment/requirements.txt
```
