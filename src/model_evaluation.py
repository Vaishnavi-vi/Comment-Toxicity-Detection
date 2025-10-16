import os
import numpy as np
import logging
from tensorflow.keras.models import load_model  # type:ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import yaml
from dvclive import Live


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "model_evaluation.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred: %s", e)
        raise


def predict():
    try:
        model = load_model("artifacts/cnn_model.h5")  
        x_train = np.load("artifacts/x_train_pad.npy")
        x_test = np.load("artifacts/x_test_pad.npy")

        y_pred_train_prob = model.predict(x_train)
        y_pred_train = (y_pred_train_prob > 0.5).astype("int")

        y_pred_test_prob = model.predict(x_test)
        y_pred_test = (y_pred_test_prob > 0.5).astype("int")

        logger.debug("Model prediction completed")
        return y_pred_test, y_pred_train
    except Exception as e:
        logger.error("Unexpected error during prediction: %s", e)
        raise


def model_evaluate(y_pred_test, y_pred_train):
    try:
        y_train = np.load("artifacts/y_train.npy")
        y_test = np.load("artifacts/y_test.npy")

        metrics_dict = {
            "accuracy_train": accuracy_score(y_train, y_pred_train),
            "accuracy_test": accuracy_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test),
            "recall": recall_score(y_test, y_pred_test),
            "f1": f1_score(y_test, y_pred_test)
        }

        logger.debug("Evaluation metrics computed")
        return metrics_dict
    except Exception as e:
        logger.error("Unexpected error during evaluation: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.debug("Metrics saved to %s", file_path)
    except Exception as e:
        logger.error("Error occurred while saving metrics: %s", e)
        raise


def main():
    try:
       params=load_params(params_path="params.yaml")
       
       y_train = np.load("artifacts/y_train.npy")
       y_test = np.load("artifacts/y_test.npy")
        
       y_pred_test,y_pred_train=predict()
       metrics=model_evaluate(y_pred_test,y_pred_train)
       
       with Live(save_dvc_exp=True,) as live:
           live.log_metric("accuracy_train",accuracy_score(y_train,y_pred_train))
           live.log_metric("accuracy_test",accuracy_score(y_test,y_pred_test))
           live.log_metric("precision",precision_score(y_test,y_pred_test))
           live.log_metric("recall",recall_score(y_test,y_pred_test))
           live.log_metric("f1",f1_score(y_test,y_pred_test))
           
           live.log_params(params)
       
       save_metrics(metrics,"artifacts\metrics.json")
       logger.debug("Model Evaluation completed successfully!")
    except Exception as e:
        logger.error("Unepected error occurred: %s",e)
        raise

if __name__ == "__main__":
    main()
