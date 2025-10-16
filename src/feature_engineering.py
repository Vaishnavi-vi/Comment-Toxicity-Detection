from tensorflow.keras.preprocessing.text import Tokenizer #type:ignore
import os
import logging
import pandas as pd
from keras.utils import pad_sequences 
import pickle
import numpy as np


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("feature_engineering")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "feature_engineering.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug("Data loaded %s",file_path)
        return df
    except FileNotFoundError as e:
        logger.error("File not found error: %s",e)
        raise
    except pd.errors.ParserError as e:
        logger.error("Failed to parse csv file: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred %s",e)
        raise
    
def token(train_data:pd.DataFrame, test_data:pd.DataFrame,num_words:int=5000,max_len:int=100,tokenizer_path:str="artifacts/tokenizer.pkl"):
    """ Apply fit_on_text on preprocessed data """
    try:
        train_data["comment_text"]=train_data["comment_text"].fillna("").astype(str)
        test_data["comment_text"]=test_data["comment_text"].fillna("").astype(str)
        tokenizer=Tokenizer(num_words=num_words,oov_token="<OOV>")
        tokenizer.fit_on_texts(train_data["comment_text"])
        tokenizer.fit_on_texts(test_data["comment_text"])

        x_train_seq=tokenizer.texts_to_sequences(train_data["comment_text"])
        x_test_seq=tokenizer.texts_to_sequences(test_data["comment_text"])
        
        x_train_pad=pad_sequences(x_train_seq,padding="post",maxlen=max_len)
        x_test_pad=pad_sequences(x_test_seq,padding="post",maxlen=max_len)
        
        
        os.makedirs(os.path.dirname(tokenizer_path),exist_ok=True)
        with open (tokenizer_path,"wb") as f:
            pickle.dump(tokenizer,f)
        np.save("artifacts/x_train_pad.npy",x_train_pad)
        np.save("artifacts/x_test_pad.npy",x_test_pad)
        logger.debug("Tokenizer fitted and and sequence padded")  
        return x_train_pad,x_test_pad
    except Exception as e:
        logger.error("Error on fit on text: %s",e)
        raise
    
def encode_labels(train_data:pd.DataFrame,test_data:pd.DataFrame):
    """ Return labels as numpy array as binary classification"""
    try:
        y_train=train_data["Toxic_label"].values
        y_test=test_data["Toxic_label"].values
        
        np.save('artifacts/y_train.npy',y_train)
        np.save('artifacts/y_test.npy',y_test)
        logger.debug("Extracted labels")
        return y_train,y_test
    except Exception as e:
        logger.error("Unexpected error occurred:%s",e)
        raise

def main():
    
    try:
        train_df=load_data("data/preprocess/train_preprocessed.csv")
        test_df=load_data("data/preprocess/test_preprocessed.csv")
        
        x_train_pad,x_test_pad=token(train_df,test_df)
        
        y_train,y_test=encode_labels(train_df,test_df)
        
        logger.info("Feature engineering completed successfully")
        logger.info(f"x_train_pad.shape:{x_train_pad.shape},x_test_pad.shape:{x_test_pad.shape}")
    except Exception as e:
        logger.error("Error in main: %s",e)
        
if __name__=="__main__":
    main()
        
    
    
    
    
    
    
    
    