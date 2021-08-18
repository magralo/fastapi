# FastAPI example

This repo intends to create and deploy a simple API on google Cloud Run. The API tries to analize financial texts to identify some asset classes and then return the polarity of each sentence that include an asset Class.

For example, when you read a financial report you might see something like the following: "I am positive on the results for the us equity in the following months", the later should be identified as an positive statement about us equity.

In financial reading there 3 kind of statements:

1. Absolute statement: This Asset Class is good. (doc_type=1 in the API result)

2. Relative statement: This asset class is better than this other asset class.

3. Non related to asset classes. (doc_type=0 in the API result)

We have to be really careful about the relative statements, that is why we only trust in the absolute statements. The API returns a JSON where you have the variable "doc_type", since we do not have a model to identify relative statments we created a doc_type=2 which refers to a sentence with more than 2 identified asset classes.

## API in action

```python
#Python Example
import requests
import json


data = {'text':'To sum up this is a bad scenario for equity as a whole'}

url =  'https://fast-y5z2fvs4lq-uc.a.run.app/analize/'

response = requests.request('POST',url,params=data, headers = {'content-type': 'application/json'})

response_dict = json.loads(response.text)
```

```console
curl -X post https://fast-y5z2fvs4lq-uc.a.run.app/analize/?text=To%20sum%20up%20this%20is%20a%20bad%20scenario%20for%20equity%20as%20a%20whole
```




## How do we split a text into sentences?

Here we use spacy and the nlp model en_core_web_lg, what we try to do is to split using all the roots in a text.

```python

roots = [token  for token in doc if token.dep_ == "ROOT" ]
for root in roots:
    token_list = [e.i for e in root.subtree]
    token_list = list(dict.fromkeys(token_list))
    token_list.sort()
    text = ' '.join([doc[i].text for i in token_list ])
    mini_doc = self.nlp(text, disable=["ner"])
    foo(mini_doc) ## Process the Span that includes only this root

```

## How do we identify the asset classes?

In an ideal world you should train a NER model... since we just want to try things, we are only indentifying some asset classes using phrase matching in spacy.

```python

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
    

```

## How do we identify the polarity in each statement?

For this task we use a transformer model trained on a financial corpus, for a better understanding of this model please refer to the original paper [of finBERT](https://arxiv.org/abs/1908.10063).




```python 

from transformers import AutoTokenizer, AutoModelForSequenceClassification

text = "Great results for emerging markets"

checkpoint = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]

```

## How to deploy using github actions?

We define a pipeline on .github/worflows/pipe.yml where we specify the steps that runs on an ubuntu machine (You can specify windows or macOS).

1. Install Python 3.8 and the dependencies to run the tests
2. Run test using pytest (We will look into this in a following section)
3. Authenticate on the google project using a service account's json that is stored as a secret.
4. Build the Dockerfile of the service. (specifying the envioronment)
5. Deploy the service on cloudrun. (specifying the envioronment)

Why use github actions? Free CI/CD

## pytest

pytest allow us to create in some simple steps the unit tests for our python application. FastAPI has a pretty easy integration with this package. When we run the pytest comand what it does is just run the test_main.py file with all the functions defined in it that starts with "test_".

The only difference that we have between an usual test of a package and the testing of an fastAPI API is that we have to create a client which is an instance of our API server:


```python 

from main import app


client = TestClient(app)


def test_readyness():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'hello':'worldhgjnsa'}


def test_noAC():
    data = {"text":"Hola Mundo"}
    response = client.post("/analize/",params = data)
    resp = json.loads(response.json())
    assert response.status_code == 200
    assert resp["AssetClass"] == ["None"]

```

To create any test of the API endpoints and methods you can use the client object an use all methods as in the package requests. For a more detailed explanation please refer to the  [fastAPI docs](https://fastapi.tiangolo.com/tutorial/testing/).


