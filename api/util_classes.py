import numpy as np
import torch
import pandas as pd
import transformers
import spacy 

class SentModel():
    def __init__(self,tokenizer,model,nlp):
        self.model = model
        self.tokenizer = tokenizer
        self.nlp = nlp 
    
    def get_sent1(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
        return predictions
    
    def get_sents (self,text):
        doc = self.nlp(text, disable=["ner"])
        deliver = []
        roots = [token  for token in doc if token.dep_ == "ROOT" ]
        deliver = []
        for root in roots:
            token_list = [e.i for e in root.subtree]
            token_list = list(dict.fromkeys(token_list))
            token_list.sort()
            text = ' '.join([doc[i].text for i in token_list ])
            mini_doc = self.nlp(text, disable=["ner"])
            doc_type = 0 # This means no identified asset class
            n_assets = len (mini_doc.ents)
            sentiment = np.array([0,0,0])
            if n_assets == 1:
                doc_type = 1 # This means absolute statement.
                sentiment = self.get_sent1(text)
            elif (n_assets > 1):
                doc_type = 2 # Talking about two asset classes.

            new = [mini_doc.ents,mini_doc.text,doc_type,sentiment[0],sentiment[1],sentiment[2]]
            deliver.append(new)
            
        deliver = pd.DataFrame(deliver)
        
        deliver.columns = ['Asset Class','Text','doc_type','Pos','Neg','Neutral']
            
        return deliver