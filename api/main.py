from fastapi import FastAPI, Form

### Download model

from google.cloud import storage




proj_id = "lateral-vision-320622"

bucket_name = "transmodel_example"

storage_client = storage.Client(proj_id)
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob('modelx.pkl')
blob.download_to_filename('modelx.pkl')


###Import model
#import pickle
from util_classes import *
#import pathlib

#temp =   pathlib.WindowsPath
#pathlib.WindowsPath = pathlib.PosixPath

#pickle_in = open("modelx.pkl","rb")
#loaded_model =    pickle.load(pickle_in)
#pathlib.WindowsPath = temp 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch 

checkpoint = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

import spacy

nlp = spacy.load("en_core_web_lg")

ruler = nlp.add_pipe("entity_ruler")

patterns = [{"label": "AC", "pattern": [{"LOWER":"equities"}], "id": "equity"},
            {"label": "AC", "pattern": "equity", "id": "equity"},
            {"label": "AC", "pattern": [{"LOWER":"us"},{"LOWER":"equities"}], "id": "us-equity"},
            {"label": "AC", "pattern": [{"LOWER":"us"},{"LOWER":"equity"}], "id": "us-equity"},
            {"label": "AC", "pattern": [{"LOWER":"american"},{"LOWER":"equities"}], "id": "us-equity"},
            {"label": "AC", "pattern": [{"LOWER":"american"},{"LOWER":"equity"}], "id": "us-equity"},
            {"label": "AC", "pattern": [{"LOWER":"european"},{"LOWER":"equities"}], "id": "eu-equity"},
            {"label": "AC", "pattern": [{"LOWER":"european"},{"LOWER":"equity"}], "id": "eu-equity"},
            {"label": "AC", "pattern": [{"LOWER":"eu"},{"LOWER":"equities"}], "id": "eu-equity"},
            {"label": "AC", "pattern": [{"LOWER":"eu"},{"LOWER":"equity"}], "id": "eu-equity"},
            {"label": "AC", "pattern": [{"LOWER":"japan"},{"LOWER":"equities"}], "id": "jp-equity"},
            {"label": "AC", "pattern": [{"LOWER":"japan"},{"LOWER":"equity"}], "id": "jp-equity"},
            {"label": "AC", "pattern": [{"LOWER":"japanesse"},{"LOWER":"equities"}], "id": "jp-equity"},
            {"label": "AC", "pattern": [{"LOWER":"japanesse"},{"LOWER":"equity"}], "id": "jp-equity"},
            {"label": "AC", "pattern": [{"LOWER":"emerging"},{"LOWER":"equities"}], "id": "emer-equity"},
            {"label": "AC", "pattern": [{"LOWER":"emerging"},{"LOWER":"equity"}], "id":"emer-equity"},
            {"label": "AC", "pattern": [{"LOWER":"emerging"},{"LOWER":"markets"}], "id": "emer-equity"},
            {"label": "AC", "pattern": [{"LOWER":"chinnese"},{"LOWER":"equity"}], "id":"china-equity"},
            {"label": "AC", "pattern": [{"LOWER":"chinnese"},{"LOWER":"equities"}], "id": "china-equity"},
            {"label": "AC", "pattern": [{"LOWER": "fixed"}, {"LOWER": "income"}], "id": "fixed'income"},
            {"label": "AC", "pattern": [{"LOWER": "fi"}], "id": "fixed'income"},
            {"label": "AC", "pattern": [{"LOWER": "commodity"}], "id": "commodity"},
            {"label": "AC", "pattern": [{"LOWER": "commodities"}], "id": "commodity"}]

ruler.add_patterns(patterns)
    
    
    
final = SentModel(tokenizer=tokenizer,model = model,nlp =nlp)



###
app = FastAPI()

@app.get('/')
async def hello_world():
    return {'hello':'worldhgjnsa'}


@app.post("/analize/")
async def sentiments(text):
    return loaded_model.get_sents(text).to_json(default_handler=str,orient="index")



## deploy local conda activate fastapi 
## deploy local uvicorn main:app --reload 

#curl -X GET -d '{"text": "hello word"}' -H "Content-type: application/json" http://127.0.0.1:8000/analize
# curl -X GET http://127.0.0.1:8000/analize/?text=i%20like%20emerging%20markets