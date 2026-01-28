# ML Integration

Practice repository for integrating machine learning model pipelines into real-world usage scenarios. The goal is to move beyond notebook experimentation toward production-ready inference—accepting user input and hopefully building experimental frontends, perhaps with streamlit or gradio.

## Focus

The specific dataset and prediction task here are secondary, so I used the infamous customer churn dataset. The main emphasis is on:

- Building reusable preprocessing and model pipelines
- Hyperparameter tuning with cross-validation (my first time using Randomized and Bayes CV)
- Handling class imbalance
- Structuring code for eventual deployment
- Exploring how to expose models for inference


## Local Setup Instructions/Guide

```
pip install -r requirements.txt
```

Requires Python 3.11. I recommend setting up a venv with the following steps:

```bash
python3 -m venv venv # to setup the venv

source venv/bin/activate # to activate the venv
```
## How to Use Local Inference (User Input):
1. Make sure venv is active (see above section)
2. Run the following command, making sure that a) venv is active with dependencies installed, b) you're in the correct wd, c) have the correct interpreter selected (3.11), and d) you've got the joblib model:
```bash
    python3 inference.py
```

## How to Use Frontend (w/ Gradio):
1. Make sure venv is active
2. Run the following command, and wait as it can sometimes take a minute or so to launch:
```bash
    python3 frontend.py
```

## Next Steps
- training in production?

