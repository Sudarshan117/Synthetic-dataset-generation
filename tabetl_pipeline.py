import os
import yaml
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
import openai
import re
import logging

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def config_parsing(yml_file: str) -> Dict[str, Any]:
    try:
        with open(yml_file, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Pipeline configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Error loading YAML file: {e}")
        raise

def data_processing(raw_csv: str, chunk_size: int = 30):
    try:
        for chunk in pd.read_csv(raw_csv, chunksize=chunk_size):
            yield chunk
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise

def filter_data(df: pd.DataFrame, filter_key: str) -> pd.DataFrame:
    logger.info(f"Filtering data using the keyword: {filter_key}")
    return df[df.apply(lambda row: filter_key in row.to_string(), axis=1)]

def sample_rows(df: pd.DataFrame, sample_size: int = 10) -> pd.DataFrame:
    sample_size = min(sample_size, len(df))
    return df.sample(n=sample_size)

def encode(df: pd.DataFrame) -> str:
    encoded_str = "; ".join([
        f"age: {row['age']}, workclass: {row['workclass']}, education: {row['education']}, marital-status: {row['marital-status']}, "
        f"occupation: {row['occupation']}, relationship: {row['relationship']}, race: {row['race']}, sex: {row['sex']}, "
        f"capital-gain: {row['capital-gain']}, capital-loss: {row['capital-loss']}, hours-per-week: {row['hours-per-week']}, "
        f"native-country: {row['native-country']}, income: {row['income']}"
        for _, row in df.iterrows()
    ])
    logger.info("Data encoded into string format.")
    return encoded_str

# Decode string back into a DataFrame
def decode(encoded_str: str) -> pd.DataFrame:
    rows = []
    for entry in encoded_str.split(";"):
        try:
            row = {
                'age': int(re.search(r"age: (\d+)", entry).group(1)),
                'workclass': re.search(r"workclass: ([\w\-]+)", entry).group(1),
                'education': re.search(r"education: ([\w\-]+)", entry).group(1),
                'marital-status': re.search(r"marital-status: ([\w\-]+)", entry).group(1),
                'occupation': re.search(r"occupation: ([\w\-]+)", entry).group(1),
                'relationship': re.search(r"relationship: ([\w\-]+)", entry).group(1),
                'race': re.search(r"race: ([\w\-]+)", entry).group(1),
                'sex': re.search(r"sex: ([\w\-]+)", entry).group(1),
                'capital-gain': int(re.search(r"capital-gain: (\d+)", entry).group(1)),
                'capital-loss': int(re.search(r"capital-loss: (\d+)", entry).group(1)),
                'hours-per-week': int(re.search(r"hours-per-week: (\d+)", entry).group(1)),
                'native-country': re.search(r"native-country: ([\w\-]+)", entry).group(1),
                'income': re.search(r"income: ([<>50K]+)", entry).group(1)
            }
            rows.append(row)
        except AttributeError:
            logger.warning(f"Skipping malformed entry: {entry}")
    logger.info("Data decoded back into DataFrame.")
    return pd.DataFrame(rows)

# Summarize data using OpenAI API
def summarization(chunk_text: str) -> str:
    logger.info("Generating summary using OpenAI...")
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": f"Summarize the following data: {chunk_text}"}],
        max_tokens=100,
        temperature=0.5
    )
    return respons

def re_generation(summary_text: str, row_count: int = 5) -> str:
    logger.info(f"Generating {row_count} synthetic rows using OpenAI...")
    prompt = (
        f"Generate {row_count} rows of data in the format: "
        f"'age: [age], workclass: [workclass], education: [education], marital-status: [marital-status], "
        f"occupation: [occupation], relationship: [relationship], race: [race], sex: [sex], capital-gain: [capital-gain], "
        f"capital-loss: [capital-loss], hours-per-week: [hours-per-week], native-country: [native-country], income: [income]'. "
        f"Summary: {summary_text}"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

class Pipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self, data_path: str):
        for chunk in data_processing(data_path, self.config.get("chunk_size", 30)):
            filtered = filter_data(chunk, self.config["filter_key"])
            sampled = sample_rows(filtered, self.config.get("sample_size", 5))
            encoded_text = encode(sampled)
            summary = summarization(encoded_text)
            regenerated_text = re_generation(summary, self.config.get("generated_rows", 5))
            decoded_df = decode(regenerated_text)

            # Save results
            output_path = self.config.get("output_file", "synthetic_data.csv")
            logger.info(f"Saving synthetic data to: {output_path}")
            decoded_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

def run_pipeline(config_path: str, data_path: str):
    config = config_parsing(config_path)
    pipeline = Pipeline(config)
    pipeline.run(data_path)

if __name__ == "__main__":
    run_pipeline("basic_model.yml", "adult.csv")
