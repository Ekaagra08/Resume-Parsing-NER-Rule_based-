import json
import re
import random
import logging
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score

import spacy
from spacy import displacy
!python -m spacy download en_core_web_md
import en_core_web_md
 
  
def convert_json_for_spacy(FilePath):
    try:
      converted_data = []
      lines=[]
      with open(FilePath, 'r') as f:
          lines = f.readlines()

      for line in lines:
          data = json.loads(line)
          text = data['content'].replace("\n", " ")
          entities = []
          data_annotations = data['annotation']
          if data_annotations is not None:
              for annotation in data_annotations:
                  #only a single point in text annotation.
                  point = annotation['points'][0]
                  labels = annotation['label']
                  # handle both list of labels or a single label.
                  if not isinstance(labels, list):
                      labels = [labels]

                  for label in labels:
                      point_start = point['start']
                      point_end = point['end']
                      point_text = point['text']

                      lstrip_diff = len(point_text) - len(point_text.lstrip())
                      rstrip_diff = len(point_text) - len(point_text.rstrip())
                      if lstrip_diff != 0:
                          point_start = point_start + lstrip_diff
                      if rstrip_diff != 0:
                          point_end = point_end - rstrip_diff
                      entities.append((point_start, point_end + 1 , label))
          
          converted_data.append((text, {"entities" : entities}))
      return converted_data
      
    except Exception as e:
      logging.exception("Unable to process " + FilePath + "\n" + "error = " + str(e))
      return None    

################################################################################################

def trim_entity_spans(data: list) -> list:
# removes extra white spaces from entity span to prevent overlaping
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    
    return cleaned_data
  
train_data_clean = trim_entity_spans(convert_json_for_spacy("/content/drive/MyDrive/Resume Parsing (NER + Rule-based)/traindata.json"))
  

def train_spacyNER():

  # creating blank eng-language class and add the built-in pipeline components to the pipeline
  c_nlp = spacy.blank("en")
  if 'ner' not in c_nlp.pipe_names:
    ner = c_nlp.create_pipe('ner')
    c_nlp.add_pipe(ner,last = True)

  # adding custom lables from resume
  for _, annotation in train_data_clean:
    for ent in annotation.get('entities'):
      ner.add_label(ent[2])    

  # other pipes to disabled during training
  other_pipes = [pipe for pipe in c_nlp.pipe_names if pipe != 'ner']
  with c_nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = c_nlp.begin_training()
    for itn in range(10):
      print("Statring iteration " + str(itn + 1))
      random.shuffle(train_data_clean)
      losses = {}
      for text, annotations in train_data_clean:
        c_nlp.update(
            [text],  # batch of texts
            [annotations],  # batch of annotations
            drop=0.2,  # dropout - make it harder to memorise data
            sgd=optimizer,  # callable to update weights
            losses=losses)
      print(losses)
  return c_nlp
  
nlp = train_spacyNER()

#######################################################################

#Testing
examples = convert_json_for_spacy("/content/drive/MyDrive/Resume Parsing (NER + Rule-based)/testdata.json")
tp=0
tr=0
tf=0
ta=0
c=0        
for text,annot in examples:

    f=open("resume"+str(c)+".txt","w")
    doc_to_test=nlp(text)
    d={}
    for ent in doc_to_test.ents:
        d[ent.label_]=[]
    for ent in doc_to_test.ents:
        d[ent.label_].append(ent.text)

    for i in set(d.keys()):

        f.write("\n\n")
        f.write(i +":"+"\n")
        for j in set(d[i]):
            f.write(j.replace('\n','')+"\n")
    d={}
    for ent in doc_to_test.ents:
        d[ent.label_]=[0,0,0,0,0,0]
    for ent in doc_to_test.ents:
        doc_gold_text= nlp.make_doc(text)
        gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
        y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
        y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]  
        if(d[ent.label_][0]==0):
            #f.write("For Entity "+ent.label_+"\n")   
            #f.write(classification_report(y_true, y_pred)+"\n")
            (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
            a=accuracy_score(y_true,y_pred)
            d[ent.label_][0]=1
            d[ent.label_][1]+=p
            d[ent.label_][2]+=r
            d[ent.label_][3]+=f
            d[ent.label_][4]+=a
            d[ent.label_][5]+=1
    c+=1
for i in d:
    print("\n For Entity "+i+"\n")
    print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
    print("Precision : "+str(d[i][1]/d[i][5]))
    print("Recall : "+str(d[i][2]/d[i][5]))
    print("F-score : "+str(d[i][3]/d[i][5]))

import pickle

filename = 'NER_model.pkl'
pickle.dump(nlp, open(filename, 'wb'))    