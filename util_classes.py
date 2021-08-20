import numpy as np
import torch
import pandas as pd
import transformers
import spacy 
import json

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
        deliver = dict({'AssetClass':[],'Text':[],'doc_type':[],'Pos':[],'Neg':[],'Neutral':[]})
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

                sentiment = self.get_sent1(text)
                doc_type = 1 # Talking about two asset classes.
                deliver['AssetClass'].append(mini_doc.ents[0].ent_id_)
                deliver['Text'].append(mini_doc.text)
                deliver['doc_type'].append("1")
                deliver['Pos'].append(str(sentiment[0]))
                deliver['Neg'].append(str(sentiment[1]))
                deliver['Neutral'].append(str(sentiment[2]))
                
            elif n_assets == 0:
                
                deliver['AssetClass'].append("None")
                deliver['Text'].append(mini_doc.text)
                deliver['doc_type'].append("0")
                deliver['Pos'].append("0")
                deliver['Neg'].append("0")
                deliver['Neutral'].append("0")
            elif n_assets > 1:
                sentiment = self.get_sent1(text)
                doc_type = 2 # Talking about two or more asset classes.          
                for ent in mini_doc.ents:

                    #deliver['AssetClass'].append(ent.ent_id_) #para inducir error, comentar
                    deliver['Text'].append(mini_doc.text)
                    deliver['doc_type'].append("2")
                    deliver['Pos'].append(str(sentiment[0]))
                    deliver['Neg'].append(str(sentiment[1]))
                    deliver['Neutral'].append(str(sentiment[2]))

               
        return json.dumps(deliver) 
        #return deliver

