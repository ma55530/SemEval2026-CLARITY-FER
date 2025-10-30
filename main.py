from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from Interview import Interview, Clarity


dataset = load_dataset("ailsntua/QEvasion")

trainingData = dataset["train"]
testingData = dataset["test"]

trainingInterviews = Interview(trainingData)
testingInterviews = Interview(testingData)

train_x = trainingInterviews.getQuestionAnswer()
train_y = trainingInterviews.getClarity()

print(train_y.count(Clarity.AMBIVALENT))
print(train_y.count(Clarity.CLEAR_REPLY))
print(train_y.count(Clarity.CLEAR_NON_REPLY))
test_x = testingInterviews.getQuestionAnswer()
test_y = testingInterviews.getClarity()

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

clf = SVC(class_weight='balanced')
clf.fit(train_x_vectors, train_y)

f1_score = f1_score(test_y, clf.predict(test_x_vectors), average='macro', labels=[Clarity.CLEAR_REPLY, Clarity.AMBIVALENT, Clarity.CLEAR_NON_REPLY])
print("F1 score: ", f1_score)

