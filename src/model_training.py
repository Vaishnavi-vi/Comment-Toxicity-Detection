from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Conv1D,Embedding,Dense,Dropout,GlobalMaxPooling1D #type:ignore
from keras.callbacks import EarlyStopping,ModelCheckpoint #type:ignore
import os
import logging
import pandas as pd
import numpy as np


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

def build_cnn_model(vocab_size,max_len):
    model=Sequential()
    model.add(Embedding(input_dim=vocab_size,output_dim=100,input_length=max_len))
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
        
        vocab_size=5000
        max_len=100
        model=build_cnn_model(vocab_size=vocab_size,max_len=max_len)
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
    
    

