
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

from Interview import Interview, Clarity

dataset = load_dataset("ailsntua/QEvasion")

trainingData = dataset["train"]
testingData = dataset["test"]

df = pd.DataFrame(columns=["model","f1 score","CLEAR_REPLY","AMBIVALENT","CLEAR_NON_REPLY"])

trainingInterviews = Interview(trainingData)
testingInterviews = Interview(testingData)

train_x = trainingInterviews.getQuestionAnswer()
train_y = trainingInterviews.getClarity()

test_x = testingInterviews.getQuestionAnswer()
test_y = testingInterviews.getClarity()

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

models = {
    "SVC": SVC(class_weight='balanced'),
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "LinearSVC": LinearSVC(class_weight="balanced"),
    "NaiveBayes": MultinomialNB()
}

labels = [Clarity.CLEAR_REPLY, Clarity.AMBIVALENT, Clarity.CLEAR_NON_REPLY]

for model_name, clf in models.items():
    clf.fit(train_x_vectors, train_y)
    y_pred = clf.predict(test_x_vectors)
    score = f1_score(test_y, y_pred, average="macro", labels=labels)
    cm = confusion_matrix(test_y, y_pred, labels=labels)
    df.loc[len(df)] = [
        model_name, score, cm[0 ,0], cm[1 ,1], cm[2 ,2]
    ]

df.to_csv("results.csv", index=False)
