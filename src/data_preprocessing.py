import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
import os
import re
import string
import pandas as pd

# Download NLTK data
nltk.download("stopwords")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("wordnet")

# Logging configuration
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """Transform the input text by lowercasing, tokenizing, removing stopwords, lemmatizing"""
    lam = WordNetLemmatizer()
    stopword = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only letters and spaces
    words = word_tokenize(text)
    clean_words = [lam.lemmatize(word) for word in words if word not in stopword and len(word) > 2]
    return " ".join(clean_words)

def pre_process(df: pd.DataFrame, text_column="comment_text") -> pd.DataFrame:
    """Preprocess the text column in the dataframe"""
    try:
        df[text_column] = df[text_column].apply(transform_text)
        logger.debug("Preprocess completed on column: %s", text_column)
        return df
    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error during text normalization: %s", e)
        raise

def main(text_column="comment_text", target_column="Toxic_label"):
    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Data loaded properly")

        # Transform the data
        train_preprocess = pre_process(train_data, text_column)
        test_preprocess = pre_process(test_data, text_column)

        # Store the data
        data_path = os.path.join("./data", "preprocess")
        os.makedirs(data_path, exist_ok=True)

        train_preprocess.to_csv(os.path.join(data_path, "train_preprocessed.csv"), index=False)
        test_preprocess.to_csv(os.path.join(data_path, "test_preprocessed.csv"), index=False)
        logger.debug("Preprocessed data saved to %s", data_path)

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to complete data transformation: %s", e)
        raise

if __name__ == "__main__":
    main()
