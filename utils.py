
from html import entities


​import​ ​io 
​import​ ​os 
​import​ ​re 
​import​ ​nltk 
​import​ ​pandas​ ​as​ ​pd 
​import​ ​docx2txt 
​from​ ​datetime​ ​import​ ​datetime 
​from​ ​dateutil​ ​import​ ​relativedelta 
​from​ . ​import​ ​constants​ ​as​ ​cs 
​from​ ​pdfminer​.​converter​ ​import​ ​TextConverter 
​from​ ​pdfminer​.​pdfinterp​ ​import​ ​PDFPageInterpreter 
​from​ ​pdfminer​.​pdfinterp​ ​import​ ​PDFResourceManager 
​from​ ​pdfminer​.​layout​ ​import​ ​LAParams 
​from​ ​pdfminer​.​pdfpage​ ​import​ ​PDFPage 
​from​ ​pdfminer​.​pdfparser​ ​import​ ​PDFSyntaxError 
​from​ ​nltk​.​stem​ ​import​ ​WordNetLemmatizer 
​from​ ​nltk​.​corpus​ ​import​ ​stopwords 

class resparser():
  def __init__(self,model,filepath):
    self.model = model
    self.filepath = filepath

  def parse(self):

    ''' Parses the PDF and displayes resume lables and entities '''
 
    try:
      doc = fitz.open(self.filepath)
      self.textblob = ""
      for pg in doc:
        self.textblob += str(pg.get_text())

    except:
      print("\nError in parse: Could not read the file ")

    finally:
      doc.close()
    
    nlp = self.model
    doc = nlp(self.textblob)
    COLOR = '\033[92m' #GREEN
    BOLD = '\033[1m'
    RESET = '\033[0m' #RESET COLOR

    for ent in doc.ents:
      print(f"{COLOR}{BOLD}{ent.label_.upper():{25}}{RESET}: {ent.text}")
    
  def text(self):
    ''' returns: Text format of parsed PDF resume '''
    return self.textblob

def extract_text(filepath, ext):
    if ext == ".pdf":
        return resparser(filepath) 
    else:
        return "Please input PDF"


def extract_section(text):
  txt_lines = [sent.strip() for sent in text.split("\n") ]
  entities = {}
  key=False
  for word in txt_lines:
    if len(word)==1:
      s_key = word
    else:
      s_key = set(word.lower.split()) & set(cs.RESUME_SECTIONS)
      s_key = list(s_key)[0]
    if s_key in cs.RESUME_SECTIONS:
      entities[s_key]=[]
      key =s_key
    elif key and word.strip():
      entities[key].append(word)    
  return entities

def custom_entities(custom_spacy_txt):
  entities={}
  for ent in  custom_spacy_txt.ents:
    if ent.label not in entities.keys():
      entities[ent.lable_] =[ent.text]
    else:
      entities[ent.lable_].append(ent.text)

  for key in entities.keys():
    entities[key] = list(set(entities[key]))
  return entities    

​def​ ​get_total_experience​(​experience_list​): 
​    ​''' 
​    Wrapper function to extract total months of experience from a resume 

​    :param experience_list: list of experience text extracted 
​    :return: total months of experience 
​    ''' 
​    ​exp_​ ​=​ [] 
​    ​for​ ​line​ ​in​ ​experience_list​: 
​        ​experience​ ​=​ ​re​.​search​( 
​            ​r'(?P<fmonth>\w+.\d+)\s*(\D|to)\s*(?P<smonth>\w+.\d+|present)'​, 
​            ​line​, 
​            ​re​.​I 
​        ) 
​        ​if​ ​experience​: 
​            ​exp_​.​append​(​experience​.​groups​()) 
​    ​total_exp​ ​=​ ​sum​( 
​        [​get_number_of_months_from_dates​(​i​[​0​], ​i​[​2​]) ​for​ ​i​ ​in​ ​exp_​] 
​    ) 
​    ​total_experience_in_months​ ​=​ ​total_exp 
​    ​return​ ​total_experience_in_months 


​def​ ​get_number_of_months_from_dates​(​date1​, ​date2​): 
​    ​''' 
​    Helper function to extract total months of experience from a resume 

​    :param date1: Starting date 
​    :param date2: Ending date 
​    :return: months of experience from date1 to date2 
​    ''' 
​    ​if​ ​date2​.​lower​() ​==​ ​'present'​: 
​        ​date2​ ​=​ ​datetime​.​now​().​strftime​(​'%b %Y'​) 
​    ​try​: 
​        ​if​ ​len​(​date1​.​split​()[​0​]) ​>​ ​3​: 
​            ​date1​ ​=​ ​date1​.​split​() 
​            ​date1​ ​=​ ​date1​[​0​][:​3​] ​+​ ​' '​ ​+​ ​date1​[​1​] 
​        ​if​ ​len​(​date2​.​split​()[​0​]) ​>​ ​3​: 
​            ​date2​ ​=​ ​date2​.​split​() 
​            ​date2​ ​=​ ​date2​[​0​][:​3​] ​+​ ​' '​ ​+​ ​date2​[​1​] 
​    ​except​ ​IndexError​: 
​        ​return​ ​0 
​    ​try​: 
​        ​date1​ ​=​ ​datetime​.​strptime​(​str​(​date1​), ​'%b %Y'​) 
​        ​date2​ ​=​ ​datetime​.​strptime​(​str​(​date2​), ​'%b %Y'​) 
​        ​months_of_experience​ ​=​ ​relativedelta​.​relativedelta​(​date2​, ​date1​) 
​        ​months_of_experience​ ​=​ (​months_of_experience​.​years 
​                                ​*​ ​12​ ​+​ ​months_of_experience​.​months​) 
​    ​except​ ​ValueError​: 
​        ​return​ ​0 
​    ​return​ ​months_of_experience 


​def​ ​extract_entity_sections_professional​(​text​): 
​    ​''' 
​    Helper function to extract all the raw text from sections of 
​    resume specifically for professionals 

​    :param text: Raw text of resume 
​    :return: dictionary of entities 
​    ''' 
​    ​text_split​ ​=​ [​i​.​strip​() ​for​ ​i​ ​in​ ​text​.​split​(​'​\n​'​)] 
​    ​entities​ ​=​ {} 
​    ​key​ ​=​ ​False 
​    ​for​ ​phrase​ ​in​ ​text_split​: 
​        ​if​ ​len​(​phrase​) ​==​ ​1​: 
​            ​p_key​ ​=​ ​phrase 
​        ​else​: 
​            ​p_key​ ​=​ ​set​(​phrase​.​lower​().​split​()) \ 
​                    ​&​ ​set​(​cs​.​RESUME_SECTIONS_PROFESSIONAL​) 
​        ​try​: 
​            ​p_key​ ​=​ ​list​(​p_key​)[​0​] 
​        ​except​ ​IndexError​: 
​            ​pass 
​        ​if​ ​p_key​ ​in​ ​cs​.​RESUME_SECTIONS_PROFESSIONAL​: 
​            ​entities​[​p_key​] ​=​ [] 
​            ​key​ ​=​ ​p_key 
​        ​elif​ ​key​ ​and​ ​phrase​.​strip​(): 
​            ​entities​[​key​].​append​(​phrase​) 
​    ​return​ ​entities 


​def​ ​extract_email​(​text​): 
​    ​''' 
​    Helper function to extract email id from text 

​    :param text: plain text extracted from resume file 
​    ''' 
​    ​email​ ​=​ ​re​.​findall​(​r"([^@|\s]+@[^@]+\.[^@|\s]+)"​, ​text​) 
​    ​if​ ​email​: 
​        ​try​: 
​            ​return​ ​email​[​0​].​split​()[​0​].​strip​(​';'​) 
​        ​except​ ​IndexError​: 
​            ​return​ ​None 


​def​ ​extract_name​(​nlp_text​, ​matcher​): 
​    ​''' 
​    Helper function to extract name from spacy nlp text 

​    :param nlp_text: object of `spacy.tokens.doc.Doc` 
​    :param matcher: object of `spacy.matcher.Matcher` 
​    :return: string of full name 
​    ''' 
​    ​pattern​ ​=​ [​cs​.​NAME_PATTERN​] 

​    ​matcher​.​add​(​'NAME'​, ​None​, ​*​pattern​) 

​    ​matches​ ​=​ ​matcher​(​nlp_text​) 

​    ​for​ ​_​, ​start​, ​end​ ​in​ ​matches​: 
​        ​span​ ​=​ ​nlp_text​[​start​:​end​] 
​        ​if​ ​'name'​ ​not​ ​in​ ​span​.​text​.​lower​(): 
​            ​return​ ​span​.​text 


​def​ ​extract_mobile_number​(​text​, ​custom_regex​=​None​): 
​    ​''' 
​    Helper function to extract mobile number from text 

​    :param text: plain text extracted from resume file 
​    :return: string of extracted mobile numbers 
​    ''' 
​    ​# Found this complicated regex on : 
​    ​# https://zapier.com/blog/extract-links-email-phone-regex/ 
​    ​# mob_num_regex = r'''(?:(?:\+?([1-9]|[0-9][0-9]| 
​    ​#     [0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]| 
​    ​#     [2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]| 
​    ​#     [0-9]1[02-9]|[2-9][02-8]1| 
​    ​#     [2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]| 
​    ​#     [2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{7}) 
​    ​#     (?:\s*(?:#|x\.?|ext\.?| 
​    ​#     extension)\s*(\d+))?''' 
​    ​if​ ​not​ ​custom_regex​: 
​        ​mob_num_regex​ ​=​ ​r'''(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\) 
​                        [-\.\s]*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})''' 
​        ​phone​ ​=​ ​re​.​findall​(​re​.​compile​(​mob_num_regex​), ​text​) 
​    ​else​: 
​        ​phone​ ​=​ ​re​.​findall​(​re​.​compile​(​custom_regex​), ​text​) 
​    ​if​ ​phone​: 
​        ​number​ ​=​ ​''​.​join​(​phone​[​0​]) 
​        ​return​ ​number 


​def​ ​extract_skills​(​nlp_text​, ​noun_chunks​, ​skills_file​=​None​): 
​    ​''' 
​    Helper function to extract skills from spacy nlp text 

​    :param nlp_text: object of `spacy.tokens.doc.Doc` 
​    :param noun_chunks: noun chunks extracted from nlp text 
​    :return: list of skills extracted 
​    ''' 
​    ​tokens​ ​=​ [​token​.​text​ ​for​ ​token​ ​in​ ​nlp_text​ ​if​ ​not​ ​token​.​is_stop​] 
​    ​if​ ​not​ ​skills_file​: 
​        ​data​ ​=​ ​pd​.​read_csv​( 
​            ​os​.​path​.​join​(​os​.​path​.​dirname​(​__file__​), ​'skills.csv'​) 
​        ) 
​    ​else​: 
​        ​data​ ​=​ ​pd​.​read_csv​(​skills_file​) 
​    ​skills​ ​=​ ​list​(​data​.​columns​.​values​) 
​    ​skillset​ ​=​ [] 
​    ​# check for one-grams 
​    ​for​ ​token​ ​in​ ​tokens​: 
​        ​if​ ​token​.​lower​() ​in​ ​skills​: 
​            ​skillset​.​append​(​token​) 

​    ​# check for bi-grams and tri-grams 
​    ​for​ ​token​ ​in​ ​noun_chunks​: 
​        ​token​ ​=​ ​token​.​text​.​lower​().​strip​() 
​        ​if​ ​token​ ​in​ ​skills​: 
​            ​skillset​.​append​(​token​) 
​    ​return​ [​i​.​capitalize​() ​for​ ​i​ ​in​ ​set​([​i​.​lower​() ​for​ ​i​ ​in​ ​skillset​])] 


​def​ ​cleanup​(​token​, ​lower​=​True​): 
​    ​if​ ​lower​: 
​        ​token​ ​=​ ​token​.​lower​() 
​    ​return​ ​token​.​strip​() 


​def​ ​extract_education​(​nlp_text​): 
​    ​''' 
​    Helper function to extract education from spacy nlp text 

​    :param nlp_text: object of `spacy.tokens.doc.Doc` 
​    :return: tuple of education degree and year if year if found 
​             else only returns education degree 
​    ''' 
​    ​edu​ ​=​ {} 
​    ​# Extract education degree 
​    ​try​: 
​        ​for​ ​index​, ​text​ ​in​ ​enumerate​(​nlp_text​): 
​            ​for​ ​tex​ ​in​ ​text​.​split​(): 
​                ​tex​ ​=​ ​re​.​sub​(​r'[?|$|.|!|,]'​, ​r''​, ​tex​) 
​                ​if​ ​tex​.​upper​() ​in​ ​cs​.​EDUCATION​ ​and​ ​tex​ ​not​ ​in​ ​cs​.​STOPWORDS​: 
​                    ​edu​[​tex​] ​=​ ​text​ ​+​ ​nlp_text​[​index​ ​+​ ​1​] 
​    ​except​ ​IndexError​: 
​        ​pass 

​    ​# Extract year 
​    ​education​ ​=​ [] 
​    ​for​ ​key​ ​in​ ​edu​.​keys​(): 
​        ​year​ ​=​ ​re​.​search​(​re​.​compile​(​cs​.​YEAR​), ​edu​[​key​]) 
​        ​if​ ​year​: 
​            ​education​.​append​((​key​, ​''​.​join​(​year​.​group​(​0​)))) 
​        ​else​: 
​            ​education​.​append​(​key​) 
​    ​return​ ​education 


​def​ ​extract_experience​(​resume_text​): 
​    ​''' 
​    Helper function to extract experience from resume text 

​    :param resume_text: Plain resume text 
​    :return: list of experience 
​    ''' 
​    ​wordnet_lemmatizer​ ​=​ ​WordNetLemmatizer​() 
​    ​stop_words​ ​=​ ​set​(​stopwords​.​words​(​'english'​)) 

​    ​# word tokenization 
​    ​word_tokens​ ​=​ ​nltk​.​word_tokenize​(​resume_text​) 

​    ​# remove stop words and lemmatize 
​    ​filtered_sentence​ ​=​ [ 
​            ​w​ ​for​ ​w​ ​in​ ​word_tokens​ ​if​ ​w​ ​not 
​            ​in​ ​stop_words​ ​and​ ​wordnet_lemmatizer​.​lemmatize​(​w​) 
​            ​not​ ​in​ ​stop_words 
​        ] 
​    ​sent​ ​=​ ​nltk​.​pos_tag​(​filtered_sentence​) 

​    ​# parse regex 
​    ​cp​ ​=​ ​nltk​.​RegexpParser​(​'P: {<NNP>+}'​) 
​    ​cs​ ​=​ ​cp​.​parse​(​sent​) 

​    ​# for i in cs.subtrees(filter=lambda x: x.label() == 'P'): 
​    ​#     print(i) 

​    ​test​ ​=​ [] 

​    ​for​ ​vp​ ​in​ ​list​( 
​        ​cs​.​subtrees​(​filter​=​lambda​ ​x​: ​x​.​label​() ​==​ ​'P'​) 
​    ): 
​        ​test​.​append​(​" "​.​join​([ 
​            ​i​[​0​] ​for​ ​i​ ​in​ ​vp​.​leaves​() 
​            ​if​ ​len​(​vp​.​leaves​()) ​>=​ ​2​]) 
​        ) 

​    ​# Search the word 'experience' in the chunk and 
​    ​# then print out the text after it 
​    ​x​ ​=​ [ 
​        ​x​[​x​.​lower​().​index​(​'experience'​) ​+​ ​10​:] 
​        ​for​ ​i​, ​x​ ​in​ ​enumerate​(​test​) 
​        ​if​ ​x​ ​and​ ​'experience'​ ​in​ ​x​.​lower​() 
​    ] 
​    ​return​ ​x
