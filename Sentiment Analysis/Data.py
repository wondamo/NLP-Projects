import pandas as pd

class Data:
    def __init__(self):
        data = pd.read_csv("review.csv", parse_dates=['reviewTime'])
        self.data = data[data['reviewText'].notnull()]
    def get_data(self):
        return self.data
    def get_column(self, column):
        return self.data[column]
    def add_feature(self, feature_name, func):
<<<<<<< HEAD
        self.data[feature_name]=self.get_column('reviewText').apply(func)
        return self.data
=======
        self.data[feature_name]=self.get_column('reviewText').apply(func)
>>>>>>> c5250777db4d490ddcd3cfb54c135ca38cfd4a19
