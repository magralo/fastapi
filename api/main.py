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
import pickle
from util_classes import *
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

pickle_in = open("modelx.pkl","rb")
loaded_model =    pickle.load(pickle_in)
pathlib.PosixPath = temp 
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