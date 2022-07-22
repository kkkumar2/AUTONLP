import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')

def finalpreprocess(text,language):
    return lemmatizer(stopword(preprocess(text),language))

def preprocess(text):

    ## Imports
    import re
    import string

    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text)
    text=re.compile('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+').sub('', text) 
    text=re.compile('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text)
    
    return text
  
# STOPWORD REMOVAL
def stopword(string,language):

    ## Imports  
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    # nltk.download('stopwords')

    a= [i for i in string.split() if i not in stopwords.words(language)]
    return ' '.join(a)

def lemmatizer(string):

    ## Imports  
    # import nltk
    # nltk.download('omw-1.4')
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer 

    wl = WordNetLemmatizer()
    word_pos_tags = word_tokenize(string)
    a=[wl.lemmatize(word) for word in word_pos_tags] 
    
    return " ".join(a)