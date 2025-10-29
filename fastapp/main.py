from fastapi import FastAPI
import pickle
from fastapi.responses import JSONResponse
import pandas as pd
import logging
import os
from fastapp.input_schema.user_input import user_input
from fastapp.input_schema.predict import prediction


app = FastAPI(
    title="Classify Toxic Comment",
    description="Predict the Toxicity of comment.",
    version="1.0"
)

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

#logging configration
logger=logging.getLogger("comment_toxicity")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"comment_toxicity.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
  
@app.get("/about")
def view():
    return {"message":"Track Genre Classification API is running!"}

@app.post("/predict")
def predict(input_data:user_input):
    
    try:
        
        user_dict={"comment":input_data.Comment_text}
        
        final_prediction=prediction(user_dict)
        logger.info("Prediction done successfully !!!")
        
        return JSONResponse(status_code=200,content={"result":final_prediction})
    except Exception as e:
        logger.error("Error occur:%s",e)
        return JSONResponse(status_code=500,content={"message":str(e)})
    
    
        
        
    
        