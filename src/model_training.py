from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Conv1D,Embedding,Dense,Dropout,GlobalMaxPooling1D #type:ignore
from keras.callbacks import EarlyStopping,ModelCheckpoint #type:ignore
import os
import logging
import pandas as pd
import numpy as np
import yaml

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_training")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "model_training.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
    "Load params from params.yaml file"
    try:
        with open(params_path,"r") as f:
            params=yaml.safe_load(f)
        logger.debug("Parameters retrieved %s",params_path)
        return params
    except FileNotFoundError as e:
        logger.error("File not found %s",e)
        raise
    except yaml.YAMLError as e:
        logger.error("Yaml error: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred:%s",e)
        raise

def build_cnn_model(vocab_size,output_dim,input_length):
    model=Sequential()
    model.add(Embedding(input_dim=vocab_size,output_dim=output_dim,input_length=input_length))
    model.add(Conv1D(filters=128,kernel_size=5,activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=64,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1,activation="sigmoid"))
    
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    return model

def train_model():
    try:
        #Load data
        x_train=np.load("artifacts/x_train_pad.npy")
        x_test=np.load("artifacts/x_test_pad.npy")
        y_train=np.load("artifacts/y_train.npy")
        y_test=np.load("artifacts/y_test.npy")
        
        params=load_params(params_path="params.yaml")
        vocab_size=params["model_training"]["vocab_size"]
        output_dim=params["model_training"]["output_dim"]
        input_length=params["model_training"]["max_len"]
    
        model=build_cnn_model(vocab_size,output_dim,input_length)
        logger.debug("Model run successfully !")
        
        #Early stopping
        early_stopping=EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True,mode="min")
        
        #checkpoint
        model_check=ModelCheckpoint("artifacts/cnn_model.h5",save_best_only=True,monitor="accuracy",mode="max")
        #fit the model
        history=model.fit(x_train,y_train,epochs=15,batch_size=32,validation_data=(x_test,y_test),callbacks=[early_stopping,model_check])
        
        logger.debug("Model training completed Successfully!!")
    except Exception as e:
        logger.error("Unexpected error occurred %s",e)
        raise
    
if __name__=="__main__":
    train_model()
    
    

