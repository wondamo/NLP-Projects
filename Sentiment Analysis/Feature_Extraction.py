
class Feature_extraction:
    def __init__(self, data):
        self.data = data

    def add_features(self):
        # add the length of review as a feature
        self.data.add_feature("num_char", self.char_length)
        # add the number of words as a feature
        self.data.add_feature("word_count", self.word_count)
        # add the average word length as a feature
        self.data.add_feature("avg_word_length", self.avg_word_length)
        # add the number of hashtag words as a feature
        self.data.add_feature("hashtag_count", self.hashtag_count)

        return self.data
        
    def char_length(self,string):
        return len(string)

    def word_count(self,string):
        # get the words in a string
        words = string.split()
        # return the number of words
        return len(words)

    def avg_word_length(self, string):
            # get the words in a string
            words = string.split()
            # get the length of each word in the string
            word_len = [len(word) for word in words]
            # return the average word length
            if len(words) == 0:
                return 0
            return sum(word_len)/len(words)

    def hashtag_count(self, string):
        # get the words in the string
        words = string.split()
        # get the words that have hashtags
        hashtag = [word for word in words if word.startswith('#')]
        # return number of hashtag words
        return len(hashtag)