**Synthetic Data Generation Pipeline**

This project provides a complete pipeline for generating and evaluating synthetic datasets using OpenAI’s GPT-4-turbo API. It covers data preparation, synthetic data creation, and model evaluation.

**Features**

Preprocessing & Cleaning: Handle raw datasets, filter, and sample rows.

Encoding: Convert tabular data into text prompts.

Synthetic Generation: Use GPT-4-turbo to generate new rows.

Decoding: Transform model outputs back into structured tabular format.

Visualization: Compare real vs. synthetic data distributions.

Evaluation: Assess data quality using ML models.

**⚙️ Requirements**

Python 3.8+

OpenAI API key

Dependencies (install via requirements.txt):
pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, yaml, python-dotenv

 **Outputs**

synthetic_data.csv, synthetic_data1.csv and synthetic_data2.csv → Generated synthetic datasets

visualizations → Comparison plots between real and synthetic distributions

model results → Performance metrics from ML evaluation

<img width="1365" height="516" alt="image" src="https://github.com/user-attachments/assets/6ea9549a-6a58-4b21-b62e-b04b75c76d5e" />

 **Notes**

This project was developed to explore synthetic data generation and machine learning evaluation using LLMs.





Developed as part of a project to explore synthetic data generation and machine learning evaluation.
