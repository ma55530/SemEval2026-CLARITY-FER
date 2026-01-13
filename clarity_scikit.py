import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from Interview import Interview, Clarity

# 1. Load Data
dataset = load_dataset("ailsntua/QEvasion")
trainingData = dataset["train"]
testingData = dataset["test"]

trainingInterviews = Interview(trainingData)
testingInterviews = Interview(testingData)

train_x = trainingInterviews.getQuestionAnswer()
train_y = trainingInterviews.getClarity()

test_x = testingInterviews.getQuestionAnswer()
test_y = testingInterviews.getClarity()

# 2. Vectorize
vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

# 3. Helpers for Class Weights
classes = np.unique(train_y)
# Calculate weights: "first calculate them"
calculated_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_y)
class_weights_dict = dict(zip(classes, calculated_weights))
print(f"Computed Class Weights: {class_weights_dict}")

# Generate sample weights for models that don't support class_weight in __init__ but support sample_weight in fit
sample_weights_train = compute_sample_weight(class_weight='balanced', y=train_y)

def get_classifiers(use_weights=False):
    """
    Returns a list of (name, classifier_instance, supports_fit_sample_weight)
    """
    # Helper to inject weights if supported in init
    def create_model(cls, **kwargs):
        if use_weights:
            # Check if class_weight is a valid parameter for this class
            # Note: This check is simple; most sklearn classifiers store valid params in _get_param_names() or we just know.
            # We will rely on known support for this fixed list.
            if cls in [SVC, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression]:
               kwargs['class_weight'] = class_weights_dict
        return cls(**kwargs)

    models = []
    
    # KNeighbors
    models.append(("Nearest Neighbors", KNeighborsClassifier(3), False))

    # Linear SVM
    models.append(("Linear SVM", create_model(SVC, kernel="linear", C=0.025), False))

    # RBF SVM
    models.append(("RBF SVM", create_model(SVC, gamma=2, C=1), False))

    # Decision Tree
    models.append(("Decision Tree", create_model(DecisionTreeClassifier, max_depth=5), False))

    # Random Forest
    models.append(("Random Forest", create_model(RandomForestClassifier, max_depth=5, n_estimators=10, max_features=1), False))

    # Neural Net
    models.append(("Neural Net", MLPClassifier(alpha=1, max_iter=1000), False)) 

    # AdaBoost
    models.append(("AdaBoost", AdaBoostClassifier(), False))

    # Naive Bayes (Multinomial for text)
    models.append(("Naive Bayes", MultinomialNB(), True if use_weights else False))

    # Logistic Regression
    models.append(("Logistic Regression", create_model(LogisticRegression, max_iter=2000), False))

    return models

results = []

NUM_RUNS = 5

# We will run two passes: one without explicit weights (or default), one with weights logic applied
scenarios = [("No Weights", False), ("With Weights", True)]

for scenario_name, use_weights_flag in scenarios:
    print(f"--- Running Scenario: {scenario_name} ---")
    
    # Get fresh models
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
                # Basic handling of fit
                if needs_sample_weight and use_weights_flag:
                    model.fit(train_x_vectors, train_y, sample_weight=sample_weights_train)
                else:
                    model.fit(train_x_vectors, train_y)
                
                predictions = model.predict(test_x_vectors)

                # Metrics
                acc = accuracy_score(test_y, predictions)
                prec = precision_score(test_y, predictions, average='macro', zero_division=0)
                rec = recall_score(test_y, predictions, average='macro', zero_division=0)
                f1 = f1_score(test_y, predictions, average='macro', zero_division=0)

                run_metrics["accuracy"].append(acc)
                run_metrics["precision"].append(prec)
                run_metrics["recall"].append(rec)
                run_metrics["f1_macro"].append(f1)
            
            except Exception as e:
                print(f"Error in {name} run {i}: {e}")
                # NaNs to be skipped in mean? Or just fail 0?
                # 0 is safer to show failure in results
                run_metrics["accuracy"].append(0)
                run_metrics["precision"].append(0)
                run_metrics["recall"].append(0)
                run_metrics["f1_macro"].append(0)

        # Average and Std
        avg_acc = np.mean(run_metrics["accuracy"])
        avg_prec = np.mean(run_metrics["precision"])
        avg_rec = np.mean(run_metrics["recall"])
        avg_f1 = np.mean(run_metrics["f1_macro"])
        std_f1 = np.std(run_metrics["f1_macro"])

        results.append({
            "Scenario": scenario_name,
            "Model": name,
            "Avg Accuracy": avg_acc,
            "Avg Precision": avg_prec,
            "Avg Recall": avg_rec,
            "Avg F1": avg_f1,
            "StD F1": std_f1,
            "F1 Score Display": f"{avg_f1:.4f} ± {std_f1:.4f}"
        })

# Save results
df_results = pd.DataFrame(results)
output_file = "results_clarity.csv"
df_results.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
