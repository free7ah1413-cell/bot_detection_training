# Leakage-Aware Bot and Credential-Stuffing Detection

Code for the paper "Leakage-Aware Machine Learning for Detecting Bot Activity and
Credential Stuffing Attacks in Web Server Logs".

## Data
Public VuFind log archive (public_v2.json, ~3 GB). It is not included here.
Download it and place it at `data/public_v2.json`, or set the environment
variable `PUBLIC_V2_JSON` to its path.

## Setup
    pip install -r requirements.txt

## Run
    python run_pipeline.py

Steps: 1 load and label, 2 session features, 3 splits (random, temporal,
IP-entity, User-Agent-entity), 4 five models (LR, RF, XGBoost, LightGBM,
Isolation Forest), 6 three-class task, 7 feature ablation, 8 statistical
validation (bootstrap CI, McNemar, five seeds), 5 SHAP. Results go to `output/`.

Seed: 42. Split ratios: 70/10/20. Sessions: (IP, User-Agent), 30-minute timeout.
