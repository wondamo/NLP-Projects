from Data import Data
from Text_Processing import Text_Processing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

print("Loading Data ...")
data = Data()

print("Preprocessing text data ...")
process = Text_Processing()
data = data.add_feature("processed_review", process.lemmatize)

print(data['processed_review'])

X_train, X_test, y_train, y_test = train_test_split(data['reviewText'], data['overall'], test_size=0.25, random_state=1)

# vectorizer = TfidfVectorizer()
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X_train)
X_vects = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_vect, y_train)
print(model.score(X_vects, y_test))