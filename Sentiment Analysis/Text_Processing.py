import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_sm")

class Text_Processing:
    def __init__(self):
        self.stop_words = STOP_WORDS

    def lemmatize(self, string):
        doc = nlp(string)
        lemma = [token.lemma_ for token in doc if token.lemma_.isalpha() or token.lemma_ not in self.stop_words]
        return ' '.join(lemma)

    def pos_tag(self, string):
        doc = nlp(string)
        self.pos = [(token.text, token.pos_) for token in doc]
        return [token.pos_ for token in doc]
        
    def num_propn(self, string):
        return self.pos_tag(string).count("PROPN")

    def num_noun(self, string):
        return self.pos_tag(string).count("NOUN")

    def ner_rec(self, string):
        doc = nlp(string)
        self.ner = [(ent.text, ent.label_) for ent in doc.ents]
        return [ent.label_ for ent in doc.ents]