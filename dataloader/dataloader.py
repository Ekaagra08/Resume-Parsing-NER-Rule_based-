def convert_json_for_spacy(FilePath):
   
    """ converts data into json format
    
        arguments: takes file path as input
        returns: converted data
    
    """
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

def preprocess(data: list) -> list:
    """
    removes extra white spaces from entity span to prevent overlaping
    arguments: inputs list of data ie. entities
    returns: pre-processed and clean entities
    
    """
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