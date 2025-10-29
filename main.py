from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from Interview import Interview, Clarity


dataset = load_dataset("ailsntua/QEvasion")

trainingData = dataset["train"]
testingData = dataset["test"]

trainingInterviews = Interview(trainingData)
testingInterviews = Interview(testingData)

train_x = trainingInterviews.getQuestionAnswer()
train_y = trainingInterviews.getClarity()

test_x = testingInterviews.getQuestionAnswer()
test_y = testingInterviews.getClarity()

vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

clf_logisticRegression = LogisticRegression(max_iter=2000)
clf_logisticRegression.fit(train_x_vectors, train_y)

f1_score = f1_score(test_y, clf_logisticRegression.predict(test_x_vectors), average=None, labels=[Clarity.CLEAR_REPLY, Clarity.AMBIVALENT, Clarity.CLEAR_NON_REPLY])
print("F1 score: ", f1_score)

