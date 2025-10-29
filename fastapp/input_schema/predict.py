import pandas as pd
import pickle
from fastapp.input_schema import user_input
import mlflow.xgboost
import logging
import os
import numpy as np
from tensorflow.keras.models import load_model  #type:ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("Toxic_comment_prediction")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "Toxic_comment_prediction.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#load artifacts 
try:
    with open("artifacts/tokenizer.pkl","rb") as f:
        tokenizer=pickle.load(f)
        logger.debug("Tokenizer loaded successfully !!!")
        
except Exception as e:
    logger.error("Unexpected error occurred: %s",e)
    raise

#load model

try:
    model = mlflow.keras.load_model("models:/Toxic_Comment_Classifier/1")
    logger.debug("Model loaded successfully from MLflow registry.")
except Exception as e:
    logger.warning("MLflow model not available (%s). Loading from pickle...", e)
    
    try:
       model_path="artifacts/cnn_model.h5"
       model=load_model(model_path)
       logger.debug("Model loaded Successfully!!!")
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise
    
def prediction(user_input:dict):
    try:
        comment=user_input.get("comment","").strip()
        
        logger.debug("Received comment")
        
        #tokenizer
        seq = tokenizer.texts_to_sequences([comment])
        padded_seq = pad_sequences(seq, maxlen=100, padding="post", truncating="post")

        # Predict
        probability = float(model.predict(padded_seq)[0][0])
        label = "Toxic" if probability >= 0.5 else "Not Toxic"
        
        confidence = probability if label == "Toxic" else 1 - probability

        logger.debug(f"Prediction completed. Score: {probability:.4f}, Label: {label}")

        return {
            "comment": comment,
            "toxicity_score": float(probability),
            "label": label,
            "confidence":confidence
        }

    except Exception as e:
        logger.error("Error during prediction: %s", e)
        raise
        
      