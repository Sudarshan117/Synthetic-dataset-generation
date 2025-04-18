Synthetic Data Generation Pipeline

This project implements a pipeline to generate synthetic data using OpenAI's GPT-4-turbo API. The pipeline includes:

Data preprocessing and cleaning
Filtering and sampling rows
Encoding data into text
Generating synthetic rows using GPT
Decoding generated text back into tabular format
Visualizing and evaluating the data

Requirements

Python 3.8+
OpenAI API key
Libraries:
pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, yaml, dotenv

How to Run
Create a .env file in the project root with your OpenAI API key:
OPENAI_API_KEY=your_api_key_here

pip install -r requirements.txt

Execute the Pipeline
To run the full pipeline:
python tabetl_pipeline.py

Generate Visualizations
To compare and visualize real vs. synthetic data:
python Visualization.py

Evaluate machine learning models:
python ml.py

Outputs
synthetic_data.csv: The generated synthetic dataset.
Figures
Model Performance Metrics: Displayed in the terminal output.



Developed as part of a project to explore synthetic data generation and machine learning evaluation.