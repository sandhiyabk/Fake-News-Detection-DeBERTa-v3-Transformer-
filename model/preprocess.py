import re
import string
import pandas as pd
from bs4 import BeautifulSoup

def clean_text(text):
    """
    Cleans text by removing HTML, URLs, special characters, and normalizing whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references and '#' from hashtags
    text = re.sub(r'\@\w+|\#','', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters and punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df, text_col='text', label_col='label'):
    """
    Performs preprocessing on a pandas DataFrame.
    """
    # Remove duplicates
    df = df.drop_duplicates(subset=[text_col])
    
    # Clean text column
    df[text_col] = df[text_col].apply(clean_text)
    
    # Drop empty strings after cleaning
    df = df[df[text_col].str.len() > 0]
    
    return df

def get_class_weights(df, label_col='label'):
    """
    Calculates class weights to handle imbalance.
    """
    counts = df[label_col].value_counts()
    total = len(df)
    weights = {label: total / counts[label] for label in counts.index}
    # Normalize weights so they sum to number of classes
    sum_weights = sum(weights.values())
    weights = {label: w * len(counts) / sum_weights for label, w in weights.items()}
    return weights

def augment_data(df, text_col='text', label_col='label'):
    """
    Placeholder for data augmentation. 
    In a production system, you would use libraries like 'nlpaug' 
    for back-translation or word-embedding based paraphrasing.
    """
    print("Suggestion: Install 'nlpaug' for robust paraphrasing augmentation.")
    # Simple example: duplicate some rows with minor noise (not real paraphrasing, just placeholder)
    # real_augmentation = nlpaug.augmenter.word.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
    return df
