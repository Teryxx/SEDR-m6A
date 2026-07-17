# SEDR-m6A Offline Test Package

This package contains the offline assets needed to evaluate the final 11 SEDR-m6A checkpoints.

## Layout

```text
data/          Full train/test TSV data and ERNIE head6 test caches
checkpoint/    11 final checkpoints, configs, manifest, and SpliceBERT weights
code/          Inference code and runner scripts
environment/   Training/inference environment record and requirements
```

## Run

```bash
bash code/run_test.sh
```

PowerShell:

```powershell
./code/run_test.ps1
```

The runner writes `test_results.csv` and prints `ACC`, `MCC`, `PRE`, `REC`, and `F1` for every dataset.

To smoke-test one dataset:

```bash
python code/test_final11_models.py --datasets liver --batch-size 8
```
