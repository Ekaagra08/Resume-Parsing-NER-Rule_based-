 
 ​from​ __future__ ​import​ ​unicode_literals 
 ​from​ __future__ ​import​ ​print_function 
 ​import​ ​re 
 ​import​ ​plac 
 ​import​ ​random 
 ​from​ ​pathlib​ ​import​ ​Path 
 ​import​ ​spacy 
 ​import​ ​json 
 ​import​ ​logging 
  
  
 ​# new entity label 
 ​LABEL​ ​=​ ​"COL_NAME" 

  
  
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
  
  
 ​@​plac​.​annotations​( 
 ​    ​model​=​(​"Model name. Defaults to blank 'en' model."​, ​"option"​, ​"m"​, ​str​), 
 ​    ​new_model_name​=​(​"New model name for model meta."​, ​"option"​, ​"nm"​, ​str​), 
 ​    ​output_dir​=​(​"Optional output directory"​, ​"option"​, ​"o"​, ​Path​), 
 ​    ​n_iter​=​(​"Number of training iterations"​, ​"option"​, ​"n"​, ​int​), 
 ​) 
 ​def​ ​main​( 
 ​    ​model​=​None​, 
 ​    ​new_model_name​=​"training"​, 
 ​    ​output_dir​=​'/home/omkarpathak27/Downloads/zipped/pyresparser/pyresparser'​, 
 ​    ​n_iter​=​30 
 ​): 
 ​    ​"""Set up the pipeline and entity recognizer, and train the new entity.""" 
 ​    ​random​.​seed​(​0​) 
 ​    ​if​ ​model​ ​is​ ​not​ ​None​: 
 ​        ​nlp​ ​=​ ​spacy​.​load​(​model​)  ​# load existing spaCy model 
 ​        ​print​(​"Loaded model '%s'"​ ​%​ ​model​) 
 ​    ​else​: 
 ​        ​nlp​ ​=​ ​spacy​.​blank​(​"en"​)  ​# create blank Language class 
 ​        ​print​(​"Created blank 'en' model"​) 
 ​    ​# Add entity recognizer to model if it's not in the pipeline 
 ​    ​# nlp.create_pipe works for built-ins that are registered with spaCy 
  
 ​    ​if​ ​"ner"​ ​not​ ​in​ ​nlp​.​pipe_names​: 
 ​        ​print​(​"Creating new pipe"​) 
 ​        ​ner​ ​=​ ​nlp​.​create_pipe​(​"ner"​) 
 ​        ​nlp​.​add_pipe​(​ner​, ​last​=​True​) 
  
 ​    ​# otherwise, get it, so we can add labels to it 
 ​    ​else​: 
 ​        ​ner​ ​=​ ​nlp​.​get_pipe​(​"ner"​) 
  
 ​    ​# add labels 
 ​    ​for​ ​_​, ​annotations​ ​in​ ​TRAIN_DATA​: 
 ​        ​for​ ​ent​ ​in​ ​annotations​.​get​(​'entities'​): 
 ​            ​ner​.​add_label​(​ent​[​2​]) 
  
 ​    ​# if model is None or reset_weights: 
 ​    ​#     optimizer = nlp.begin_training() 
 ​    ​# else: 
 ​    ​#     optimizer = nlp.resume_training() 
 ​    ​move_names​ ​=​ ​list​(​ner​.​move_names​) 
 ​    ​# get names of other pipes to disable them during training 
 ​    ​other_pipes​ ​=​ [​pipe​ ​for​ ​pipe​ ​in​ ​nlp​.​pipe_names​ ​if​ ​pipe​ ​!=​ ​"ner"​] 
 ​    ​with​ ​nlp​.​disable_pipes​(​*​other_pipes​):  ​# only train NER 
 ​        ​optimizer​ ​=​ ​nlp​.​begin_training​() 
 ​        ​# batch up the examples using spaCy's minibatch 
 ​        ​for​ ​itn​ ​in​ ​range​(​n_iter​): 
 ​            ​print​(​"Starting iteration "​ ​+​ ​str​(​itn​)) 
 ​            ​random​.​shuffle​(​TRAIN_DATA​) 
 ​            ​losses​ ​=​ {} 
 ​            ​for​ ​text​, ​annotations​ ​in​ ​TRAIN_DATA​: 
 ​                ​nlp​.​update​( 
 ​                    [​text​],  ​# batch of texts 
 ​                    [​annotations​],  ​# batch of annotations 
 ​                    ​drop​=​0.2​,  ​# dropout - make it harder to memorise data 
 ​                    ​sgd​=​optimizer​,  ​# callable to update weights 
 ​                    ​losses​=​losses​) 
 ​            ​print​(​"Losses"​, ​losses​) 
  
 ​    ​# test the trained model 
 ​    ​test_text​ ​=​ ​"Marathwada Mitra Mandals College of Engineering" 
 ​    ​doc​ ​=​ ​nlp​(​test_text​) 
 ​    ​print​(​"Entities in '%s'"​ ​%​ ​test_text​) 
 ​    ​for​ ​ent​ ​in​ ​doc​.​ents​: 
 ​        ​print​(​ent​.​label_​, ​ent​.​text​) 
  
 ​    ​# save model to output directory 
 ​    ​if​ ​output_dir​ ​is​ ​not​ ​None​: 
 ​        ​output_dir​ ​=​ ​Path​(​output_dir​) 
 ​        ​if​ ​not​ ​output_dir​.​exists​(): 
 ​            ​output_dir​.​mkdir​() 
 ​        ​nlp​.​meta​[​"name"​] ​=​ ​new_model_name​  ​# rename model 
 ​        ​nlp​.​to_disk​(​output_dir​) 
 ​        ​print​(​"Saved model to"​, ​output_dir​) 
  
 ​        ​# test the saved model 
 ​        ​print​(​"Loading from"​, ​output_dir​) 
 ​        ​nlp2​ ​=​ ​spacy​.​load​(​output_dir​) 
 ​        ​# Check the classes have loaded back consistently 
 ​        ​assert​ ​nlp2​.​get_pipe​(​"ner"​).​move_names​ ​==​ ​move_names 
 ​        ​doc2​ ​=​ ​nlp2​(​test_text​) 
 ​        ​for​ ​ent​ ​in​ ​doc2​.​ents​: 
 ​            ​print​(​ent​.​label_​, ​ent​.​text​) 
  
  
 ​if​ ​__name__​ ​==​ ​"__main__"​: 
 ​    ​plac​.​call​(​main​)
