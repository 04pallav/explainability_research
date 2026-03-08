# explainability_research
Repo for explainability research.

## Requirements
- **Python 3.9** (same as CI). Use `pyenv install 3.9` and `pyenv local 3.9` if needed.
- Install deps: `pip install -r requirements.txt`

## Run
```bash
python shap_stability.py --dataset california --export-tables --no-lime --shap-sample 100
# repeat for adult, titanic to fill all paper tables
```
