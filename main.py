from fastapi import FastAPI
from util_classes import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
    
    
    
model = SentModel(tokenizer=tokenizer,model = model,nlp =nlp)



###
app = FastAPI()

@app.get('/')
async def hello_world():
    return {'hello':'world'}


@app.post("/analize/")
async def sentiments(text):
    return model.get_sents(text)



## deploy local conda activate fastapi 
## deploy local uvicorn main:app --reload 

#curl -X GET -d '{"text": "hello word"}' -H "Content-type: application/json" http://127.0.0.1:8000/analize
# curl -X POST https://fastdev-y5z2fvs4lq-uc.a.run.app/analize/analize/?text=I%20like%20emerging%20markets%20for%20the%20next%20quarter.%20I%20do%20not%20like%20US%20equity,%20the%20had%20an%20awfull%20result