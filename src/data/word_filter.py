import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer as twt

class WordFilter:
    def __init__(self, filters:str=None):
        self.filters = filters

        if filters:
            self.setup_filters(filters)

    def setup_filters(self, filters):
        # setting up stopwords filtering
        if 'S' in filters:
            nltk.download('stopwords')
            self.stopwords = stopwords.words()

        if any([i in filters for i in ['N', 'V', 'A']]):
            nltk.download('averaged_perceptron_tagger')
            nltk.download('universal_tagset')

    def filter_text(self, text):
        # Setting up POS tagger if any POS is to be filtered
        
        """
        if any(i in self.filters for i in ['V', 'N', 'A', 'P']):
            tokens = twt().tokenize(text)
            tags = nltk.pos_tag(tokens, tagset = "universal")
            text = ' '.join([i for (i, POS) in tags if POS not in self.filters])
        """
        
        if self.filters == 'S': # NO stopwords
            text = ' '.join([w for w in text.split() if not w in self.stopwords])

        elif self.filters == 'NV': # noun and verbs
            tokens = twt().tokenize(text)
            tags = nltk.pos_tag(tokens, tagset = "universal")
            text = ' '.join([i for (i, POS) in tags if POS in ['NOUN', 'VERB']])
            
        elif self.filters == 'ANV': # adjective, noun and verbs
            tokens = twt().tokenize(text)
            tags = nltk.pos_tag(tokens, tagset = "universal")
            text = ' '.join([i for (i, POS) in tags if POS in ['VERB', 'NOUN', 'ADJ']])

        elif self.filters == 'AD': # adverbs and adjectives
            tokens = twt().tokenize(text)
            tags = nltk.pos_tag(tokens, tagset = "universal")
            text = ' '.join([i for (i, POS) in tags if POS in ['ADJ', 'ADV']])

        return text

    def __call__(self, text:str):
        return self.filter_text(text)