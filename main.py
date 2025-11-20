from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from imblearn.over_sampling import SMOTE

from Interview import Interview, Clarity

dataset = load_dataset("ailsntua/QEvasion")

trainingData = dataset["train"]
testingData = dataset["test"]

df = pd.DataFrame(columns=[
    "model",
    "binary_f1",
    "3class_f1",
    "CLEAR_REPLY correct",
    "AMBIVALENT correct",
    "CLEAR_NON_REPLY correct"
])

trainingInterviews = Interview(trainingData)
testingInterviews = Interview(testingData)

train_x = trainingInterviews.getQuestionAnswer()
train_y = trainingInterviews.getClarity()

test_x = testingInterviews.getQuestionAnswer()
test_y = testingInterviews.getClarity()

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

def to_binary(labels):
    return [1 if l == Clarity.CLEAR_REPLY else 0 for l in labels]

train_y_bin = to_binary(train_y)
test_y_bin = to_binary(test_y)

models = {
    "SVC": SVC(class_weight='balanced'),
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "LinearSVC": LinearSVC(class_weight="balanced"),
    "NaiveBayes": MultinomialNB()
}

three_class_labels = [Clarity.CLEAR_REPLY, Clarity.AMBIVALENT, Clarity.CLEAR_NON_REPLY]

for model_name, clf in models.items():
    clf.fit(train_x_vectors, train_y_bin)
    pred_bin = clf.predict(test_x_vectors)
    binary_f1 = f1_score(test_y_bin, pred_bin)
    mask_non_clear = pred_bin == 0
    train_x_3 = train_x_vectors
    train_y_3 = train_y
    smote = SMOTE()
    train_x_3_balanced, train_y_3_balanced = smote.fit_resample(train_x_3, train_y_3)
    clf_3 = models[model_name]  
    clf_3.fit(train_x_3_balanced, train_y_3_balanced)
    pred_3class = clf_3.predict(test_x_vectors)
    final_pred = []
    for b, p3 in zip(pred_bin, pred_3class):
        if b == 1:
            final_pred.append(Clarity.CLEAR_REPLY)
        else:
            final_pred.append(p3)
    f1_macro = f1_score(test_y, final_pred, average="macro", labels=three_class_labels)
    cm = confusion_matrix(test_y, final_pred, labels=three_class_labels)
    df.loc[len(df)] = [
        model_name,
        binary_f1,
        f1_macro,
        cm[0,0], cm[1,1], cm[2,2]
    ]

df.to_csv("results_double_classifier_smote.csv", index=False)
