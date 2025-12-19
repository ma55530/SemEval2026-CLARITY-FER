import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

MODEL_CHECKPOINT = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
BATCH_SIZE = 8
GRAD_ACCUMULATION = 4
LEARNING_RATE = 3e-5
EPOCHS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("ailsntua/QEvasion")

clarity_mapping = {"Clear Reply": 0, "Ambivalent": 1, "Clear Non-Reply": 2}

train_labels = [x for x in dataset['train']['evasion_label'] if x is not None]
test_labels = []
for col in ['annotator1', 'annotator2', 'annotator3']:
    test_labels.extend([x for x in dataset['test'][col] if x is not None])

unique_evasion_labels = sorted(list(set(train_labels + test_labels)))
evasion_encoder = LabelEncoder()
evasion_encoder.fit(unique_evasion_labels)

def get_majority_vote(example):
    votes = []
    for col in ['annotator1', 'annotator2', 'annotator3']:
        val = example.get(col)
        if val in unique_evasion_labels:
            votes.append(val)
    if not votes: return None
    return Counter(votes).most_common(1)[0][0]

def preprocess_data(example):
    example["labels_clarity"] = clarity_mapping.get(example["clarity_label"], -1)
    
    final_evasion_str = None
    if example.get("evasion_label") in unique_evasion_labels:
        final_evasion_str = example["evasion_label"]
    else:
        final_evasion_str = get_majority_vote(example)
        
    if final_evasion_str:
        example["labels_evasion"] = int(evasion_encoder.transform([final_evasion_str])[0])
    else:
        example["labels_evasion"] = -100 
    return example

dataset = dataset.map(preprocess_data)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    return tokenizer(
        examples["interview_question"],
        examples["interview_answer"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels_clarity", "labels_evasion"])

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

train_y = np.array(tokenized_datasets["train"]["labels_evasion"])
train_y = train_y[train_y != -100]
present_classes = np.unique(train_y)
computed_weights = compute_class_weight('balanced', classes=present_classes, y=train_y)

final_weights = np.ones(len(unique_evasion_labels))
for cls, w in zip(present_classes, computed_weights):
    final_weights[cls] = w

weights_tensor = torch.tensor(final_weights, dtype=torch.float).to(DEVICE)

class MultiTaskDeberta(PreTrainedModel):
    def __init__(self, config, evasion_weights):
        super().__init__(config)
        self.deberta = AutoModel.from_config(config)
        self.classifier_clarity = nn.Linear(config.hidden_size, 3)
        self.classifier_evasion = nn.Linear(config.hidden_size, len(unique_evasion_labels))
        
        self.focal_loss_evasion = FocalLoss(alpha=evasion_weights, gamma=2.0)
        self.loss_clarity = nn.CrossEntropyLoss()
        
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels_clarity=None, labels_evasion=None, **kwargs):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        logits_clarity = self.classifier_clarity(cls_token)
        logits_evasion = self.classifier_evasion(cls_token)
        
        loss = None
        if labels_clarity is not None and labels_evasion is not None:
            loss_c = self.loss_clarity(logits_clarity, labels_clarity)
            loss_e = self.focal_loss_evasion(logits_evasion, labels_evasion)
            
            loss = (1.0 * loss_c) + (4.0 * loss_e)

        return (loss, logits_clarity, logits_evasion) if loss is not None else (logits_clarity, logits_evasion)

config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
model = MultiTaskDeberta.from_pretrained(MODEL_CHECKPOINT, config=config, evasion_weights=weights_tensor).to(DEVICE)

for i in range(6):
    for param in model.deberta.encoder.layer[i].parameters():
        param.requires_grad = False
for param in model.deberta.embeddings.parameters():
    param.requires_grad = False

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels_clarity=inputs.get("labels_clarity"),
            labels_evasion=inputs.get("labels_evasion")
        )
        return (outputs[0], outputs) if return_outputs else outputs[0]

def compute_multitask_metrics(eval_pred):
    (logits_cl, logits_ev), (labels_cl, labels_ev) = eval_pred
    preds_cl = np.argmax(logits_cl, axis=-1)
    preds_ev = np.argmax(logits_ev, axis=-1)
    
    mask = labels_ev != -100
    if mask.sum() == 0: return {"combined_f1": 0}

    evasion_f1 = f1_score(labels_ev[mask], preds_ev[mask], average="macro", zero_division=0)
    clarity_f1 = f1_score(labels_cl, preds_cl, average="macro", zero_division=0)
    
    return {
        "clarity_f1": clarity_f1, 
        "evasion_f1": evasion_f1, 
        "combined_f1": (clarity_f1 + evasion_f1) / 2
    }

training_args = TrainingArguments(
    output_dir="./optimized_multitask_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="evasion_f1",
    remove_unused_columns=False,
    report_to="none"
)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_multitask_metrics,
)

trainer.train()

preds_output = trainer.predict(tokenized_datasets["test"])
y_pred = np.argmax(preds_output.predictions[1], axis=-1)
y_true = preds_output.label_ids[1]
mask = y_true != -100
print(classification_report(y_true[mask], y_pred[mask], target_names=evasion_encoder.classes_, zero_division=0))

cm = confusion_matrix(y_true[mask], y_pred[mask])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=evasion_encoder.classes_, 
            yticklabels=evasion_encoder.classes_)
plt.title('Evasion Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
