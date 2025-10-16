import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import logging
import yaml

#Ensure the log_dir exists
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

#logging configration
logger=logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"data_ingestion.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')

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
    

def load_data(data_url:str)->pd.DataFrame:
    """Load data from csv file"""
    try:
        df=pd.read_csv(data_url,engine="python",on_bad_lines="skip")
        df=df.sample(100000,random_state=42)
        logger.debug("Data loaded from %s",data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file :%s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred %s",e) 
        raise
    
def pre_process(df:pd.DataFrame)->pd.DataFrame:
    """ Preprocess the Data"""
    try:
        toxic_cols=['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']
        df["Toxic_label"]=df[toxic_cols].any(axis=1).astype(int)
        df.drop(['id', 'toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate'],axis=1,inplace=True)
        logger.debug("Data Preprocessing Completed")
        return df
    except KeyError as e:
        logger.error("Missing Columns in the Data Frame %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred %s",e)
        raise
    
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save train and test dataset"""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        
        # Create both 'data' and 'data/raw' directories if they don’t exist
        os.makedirs(raw_data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occurred %s", e)
        raise

    
def main():
    try:
        params=load_params(params_path="params.yaml")
        test_size=params["data_ingestion"]["test_size"]
        data_url="C:\\Users\\Dell\\OneDrive - Havells\\Desktop\\Comment-Toxicity-Detection\\Experiments\\dataset.csv"
        df=load_data(data_url=data_url)
        final_df=pre_process(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data=train_data,test_data=test_data,data_path="./data")
    except Exception as e:
        logger.error("Unexpected Error occurred %s",e)
        raise

if __name__=="__main__":
    main()
        
    