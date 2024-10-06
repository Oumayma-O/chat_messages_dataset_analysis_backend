from typing import Dict
import pandas as pd

def count_languages(df: pd.DataFrame):
    return df['lang'].nunique()

def lang_distribution(df: pd.DataFrame):
    return df['lang'].value_counts().to_dict()


def count_null_lang(df: pd.DataFrame) -> dict:
    null_count = df['lang'].isnull().sum()
    return {
        "count": null_count,
        "percentage": percentage_null_lang(df)["percentage"]
    }

def count_null_toxicity(df: pd.DataFrame) -> dict:
    null_count = df['detoxify'].isnull().sum()
    return {
        "count": null_count,
        "percentage": percentage_null_toxicity(df)["percentage"]
    }

def percentage_null_lang(df: pd.DataFrame) -> dict:
    total_rows = df.shape[0]
    null_count = df['lang'].isnull().sum()
    percentage = (null_count / total_rows) if total_rows > 0 else 0.0
    return {"count": null_count, "percentage": percentage}

def percentage_null_toxicity(df: pd.DataFrame) -> dict:
    total_rows = df.shape[0]
    null_count = df['detoxify'].isnull().sum()
    percentage = (null_count / total_rows)  if total_rows > 0 else 0.0
    return {"count": null_count, "percentage": percentage}

def toxicity_distribution(df: pd.DataFrame) -> dict:
    toxicity_scores = {
        "toxicity": [],
        "severe_toxicity": [],
        "obscene": [],
        "identity_attack": [],
        "insult": [],
        "threat": [],
        "sexual_explicit": [],
    }
    
    threshold = 0.01
    
    for _, row in df.iterrows():
        if pd.notnull(row['detoxify']):  
            toxicity_data = row['detoxify'] 
            for key in toxicity_scores.keys():
                score = toxicity_data.get(key, 0)
                rounded_score = score if score >= threshold else 0
                toxicity_scores[key].append(rounded_score)
    
    average_distribution = {key: sum(values) / len(values) for key, values in toxicity_scores.items() if values}

    return average_distribution
    
