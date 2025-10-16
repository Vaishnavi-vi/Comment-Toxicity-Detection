import os
import numpy as np
import logging
from tensorflow.keras.models import load_model #type:ignore
from sklearn.metrics import  accuracy_score,precision_score,recall_score,f1_score
import json
import yaml

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "model_evaluation.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def predict():
    try:
        model=load_model("artifacts/cnn_model.h5")
        x_train=np.load("artifacts/x_train_pad.npy")
        x_test=np.load("artifacts/x_test_pad.npy")
        
        
        y_pred_train_prob=model.predict(x_train)
        y_pred_train=(y_pred_train_prob>0.5).astype("int")
        
        y_pred_test_prob=model.predict(x_test)
        y_pred_test=(y_pred_test_prob>0.5).astype("int")
        
        logger.debug("Model Predict ")
        return y_pred_test,y_pred_train
    except Exception as e:
        logger.error("Unexpected Error occurred: %s",e)
        raise
    
def model_evaluate(y_pred_test,y_pred_train):
    try:
        y_train=np.load("artifacts/y_train.npy")
        y_test=np.load("artifacts/y_test.npy")
        
        accuracy_test=accuracy_score(y_test,y_pred_test)
        accuracy_train=accuracy_score(y_train,y_pred_train)
        
        precision=precision_score(y_test,y_pred_test)
        recall=recall_score(y_test,y_pred_test)
        f1=f1_score(y_test,y_pred_test)
        
        metrics_dict={
            "accuracy_train":accuracy_train,
            "accuracy_test":accuracy_test,
            "precision":precision,
            "recall":recall,
            "f1":f1
        }
        
        logger.debug("Evaluate model : %s")
        return metrics_dict
    except Exception as e:
        logger.error("Unexpected Error occurred: %s",e)
        raise
    
def save_metrics(metrics:dict,file_path:str) ->None:
    """ Save evaluation metrics to a json file """
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as f:
            json.dump(metrics,f,indent=4)
        logger.debug("Metrics saved : %s",file_path)
    except Exception as e:
        logger.error("Error occurred while saving the metrics: %s",e)
        raise
    
def main():
    try:
        y_pred_test,y_pred_train=predict()
        metrics=model_evaluate(y_pred_test,y_pred_train)
        save_metrics(metrics,"artifacts/metrics.json")
        logger.debug("Successful evaluation occur")
    except Exception as e:
        logger.error("Unexpected Error Occurred")
        raise
   
    
    
if __name__=="__main__":
    main()