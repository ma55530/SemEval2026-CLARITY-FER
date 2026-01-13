import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from collections import Counter

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from Interview import Interview

# 1. Load Data
print("Loading data...")
dataset = load_dataset("ailsntua/QEvasion")
trainingData = dataset["train"]
testingData = dataset["test"]

trainingInterviews = Interview(trainingData)
testingInterviews = Interview(testingData)

train_x = trainingInterviews.getQuestionAnswer()
train_y = trainingInterviews.getEvasion()

test_x = testingInterviews.getQuestionAnswer()
test_y_annotators = testingInterviews.getAnnotators()

# Create a cleaned list of annotators for validation
cleaned_test_annotators = []
for anns in test_y_annotators:
    # Filter out None/Empty and duplicates
    valid_anns = list(set([a for a in anns if a]))
    if not valid_anns:
        # Fallback if no annotators exist (should not happen in valid data)
        valid_anns = ["Unknown"]
    cleaned_test_annotators.append(valid_anns)

# 2. Vectorize
print("Vectorizing...")
vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

# 3. Helpers for Class Weights
print("Computing weights...")
# Let's filter None in train_y just in case for weight computation
train_y_clean = [y if y else "Unknown" for y in train_y]
classes = np.unique(train_y_clean)

calculated_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_y_clean)
class_weights_dict = dict(zip(classes, calculated_weights))
print(f"Computed Class Weights: {class_weights_dict}")

sample_weights_train = compute_sample_weight(class_weight='balanced', y=train_y_clean)

def get_classifiers(use_weights=False):
    """
    Returns a list of (name, classifier_instance, supports_fit_sample_weight)
    """
    def create_model(cls, **kwargs):
        if use_weights:
            if cls in [SVC, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression]:
               kwargs['class_weight'] = class_weights_dict
        return cls(**kwargs)

    models = []
    models.append(("Nearest Neighbors", KNeighborsClassifier(3), False))
    models.append(("Linear SVM", create_model(SVC, kernel="linear", C=0.025), False))
    models.append(("RBF SVM", create_model(SVC, gamma=2, C=1), False))
    models.append(("Decision Tree", create_model(DecisionTreeClassifier, max_depth=5), False))
    models.append(("Random Forest", create_model(RandomForestClassifier, max_depth=5, n_estimators=10, max_features=1), False))
    models.append(("Neural Net", MLPClassifier(alpha=1, max_iter=1000), False)) 
    models.append(("AdaBoost", AdaBoostClassifier(), False))
    models.append(("Naive Bayes", MultinomialNB(), True if use_weights else False))
    models.append(("Logistic Regression", create_model(LogisticRegression, max_iter=2000), False))

    return models

results = []

NUM_RUNS = 5
scenarios = [("No Weights", False), ("With Weights", True)]

def calculate_relaxed_metrics(predictions, annotator_lists, class_list):
    """
    Calculates Accuracy, Precision, Recall, and F1 based on 'Any Match' relaxed criteria.
    - Accuracy: Correct if prediction is in the set of annotators.
    - Precision/Recall/F1: Calculated per class and macro-averaged.
      - TP(c): Predicted c, and c is in annotators.
      - FP(c): Predicted c, and c is NOT in annotators.
      - FN(c): Predicted NOT c (and prediction was INCORRECT for the instance), and c WAS in annotators.
        (Note: If prediction was different but VALID (e.g. pred=A, anns={A,B}), it counts as TP(A) and NOT FN(B)).
    """
    tp = {c: 0 for c in class_list}
    fp = {c: 0 for c in class_list}
    fn = {c: 0 for c in class_list}
    
    correct_count = 0
    total = len(predictions)
    
    for pred, anns in zip(predictions, annotator_lists):
        if pred in anns:
            # Correct Prediction
            correct_count += 1
            if pred in tp:
                tp[pred] += 1
            # In a relaxed setting, a correct prediction validates the instance.
            # We do NOT count FN for other valid labels in 'anns'.
        else:
            # Incorrect Prediction
            if pred in fp:
                fp[pred] += 1
            # It implies we missed the opportunity for the actual valid labels
            for ann in anns:
                if ann in fn:
                    fn[ann] += 1
                    
    accuracy = correct_count / total if total > 0 else 0
    
    f1_sum = 0
    prec_sum = 0
    rec_sum = 0
    valid_classes = 0
    
    for c in class_list:
        # Only consider classes that appeared? Or all classes in class_list?
        # Using class_list ensures we cover all known classes.
        
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
        
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
            
        f1_sum += f1
        prec_sum += precision
        rec_sum += recall
        valid_classes += 1
        
    avg_f1 = f1_sum / valid_classes if valid_classes > 0 else 0
    avg_prec = prec_sum / valid_classes if valid_classes > 0 else 0
    avg_rec = rec_sum / valid_classes if valid_classes > 0 else 0
    
    return accuracy, avg_prec, avg_rec, avg_f1

for scenario_name, use_weights_flag in scenarios:
    print(f"--- Running Scenario: {scenario_name} ---")
    
    models_list = get_classifiers(use_weights=use_weights_flag)
    
    for name, model, needs_sample_weight in models_list:
        print(f"Processing {name}...")
        
        run_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_macro": []
        }

        for i in range(NUM_RUNS):
            try:
                if needs_sample_weight and use_weights_flag:
                    model.fit(train_x_vectors, train_y_clean, sample_weight=sample_weights_train)
                else:
                    model.fit(train_x_vectors, train_y_clean)
                
                predictions = model.predict(test_x_vectors)

                # Relaxed Metrics
                acc, prec, rec, f1 = calculate_relaxed_metrics(predictions, cleaned_test_annotators, classes)

                run_metrics["accuracy"].append(acc)
                run_metrics["precision"].append(prec)
                run_metrics["recall"].append(rec)
                run_metrics["f1_macro"].append(f1)
            
            except Exception as e:
                print(f"Error in {name} run {i}: {e}")
                run_metrics["accuracy"].append(0)
                run_metrics["precision"].append(0)
                run_metrics["recall"].append(0)
                run_metrics["f1_macro"].append(0)

        # Average and Std
        avg_acc = np.mean(run_metrics["accuracy"])
        avg_prec = np.mean(run_metrics["precision"])
        # avg_rec = np.mean(run_metrics["recall"]) # Not saving recall in CSV shown previously, but calculated
        avg_rec = np.mean(run_metrics["recall"])
        avg_f1 = np.mean(run_metrics["f1_macro"])
        std_f1 = np.std(run_metrics["f1_macro"])

        results.append({
            "Scenario": scenario_name,
            "Model": name,
            "Avg Accuracy (Any Match)": avg_acc,
            "Avg Precision (Relaxed)": avg_prec,
            "Avg Recall (Relaxed)": avg_rec,
            "Avg F1 (Relaxed)": avg_f1,
            "StD F1": std_f1,
            "F1 Score Display": f"{avg_f1:.4f} ± {std_f1:.4f}"
        })

# Save results
df_results = pd.DataFrame(results)
output_file = "results_evasion.csv"
df_results.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
