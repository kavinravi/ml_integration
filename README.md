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

## Next Steps

- add dynamic inference (user input?)
- build out tentative frontend (gradio/streamlit)
- training in production?

