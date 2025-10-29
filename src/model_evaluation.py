import os
import numpy as np
import logging
from tensorflow.keras.models import load_model #type:ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve,ConfusionMatrixDisplay
import json
import yaml
import matplotlib.pyplot as plt
import mlflow
import yaml



mlflow.set_tracking_uri("file:///C:/Users/Dell/OneDrive - Havells/Desktop/Comment-Toxicity-Detection/mlruns")

mlflow.set_experiment("CNN_Toxicity_Classification")


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
    except Exception as e:
        logger.error("Error loading params: %s", e)
        raise


def predict():
    model = load_model("artifacts/cnn_model.h5")  
    x_train = np.load("artifacts/x_train_pad.npy")
    x_test = np.load("artifacts/x_test_pad.npy")

    y_pred_train_prob = model.predict(x_train)
    y_pred_train = (y_pred_train_prob > 0.5).astype("int")

    y_pred_test_prob = model.predict(x_test)
    y_pred_test = (y_pred_test_prob > 0.5).astype("int")

    logger.debug("Model prediction completed")
    return y_pred_test, y_pred_train, y_pred_test_prob, y_pred_train_prob,model


def model_evaluate(y_pred_test, y_pred_train, y_test, y_train):
    metrics_dict = {
        "accuracy_train": accuracy_score(y_train, y_pred_train),
        "accuracy_test": accuracy_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test),
        "recall": recall_score(y_test, y_pred_test),
        "f1": f1_score(y_test, y_pred_test),
        "roc_auc": roc_auc_score(y_test, y_pred_test)
    }
    logger.debug("Evaluation metrics computed")
    return metrics_dict


def plot_confusion_matrix(y_true, y_pred, file_path):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap=plt.cm.Blues)  # you can choose other colormaps
        plt.title("Confusion Matrix")
        plt.grid(False)
        plt.savefig(file_path)
        plt.close()
        logger.debug("Confusion matrix is saved to: %s",file_path)
    except Exception as e:
        logger.error("Error occurred %s",e)
        raise

def plot_roc_curve(y_true, y_prob, file_path):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, (y_prob>0.5).astype(int))
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()
        logger.debug("AUC-roc score saved to :%s",file_path)
    except Exception as e:
        logger.error("Unexpected error occurred: %s",e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.debug("Metrics saved to %s", file_path)


def main():
    try:
        params = load_params("params.yaml")

        y_train = np.load("artifacts/y_train.npy")
        y_test = np.load("artifacts/y_test.npy")

        y_pred_test, y_pred_train, y_pred_test_prob, y_pred_train_prob,model = predict()
        metrics = model_evaluate(y_pred_test, y_pred_train, y_test, y_train)

        # MLflow logging
        with mlflow.start_run():
            # log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            # log parameters
            for key, value in params["model_training"].items():
                mlflow.log_param(key, value)
            # save metrics.json
            metrics_path = "artifacts/metrics.json"
            save_metrics(metrics, metrics_path)
            mlflow.log_artifact(metrics_path)

            # Confusion matrix plot
            cm_path = "artifacts/confusion_matrix.png"
            plot_confusion_matrix(y_test, y_pred_test, cm_path)
            mlflow.log_artifact(cm_path)

            # ROC curve plot
            roc_path = "artifacts/roc_curve.png"
            plot_roc_curve(y_test, y_pred_test_prob, roc_path)
            mlflow.log_artifact(roc_path)
            
            #log the trained model
            mlflow.keras.log_model(model,artifact_path="models",registered_model_name="Toxic_Comment_Classifier")

        logger.debug("Model evaluation and MLflow logging completed successfully!")
    except Exception as e:
        logger.error("Unexpected error occurred: %s", e)
        raise


if __name__ == "__main__":
    main()

