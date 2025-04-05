import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Load the dataset
df= pd.read_csv('fb_labeled_balanced.csv')  # Load your dataset
df


df[df['Sentiment'] == 'N']['Comment']


df.head()


df.isnull().sum()


df.duplicated().sum()


df.drop_duplicates(inplace=True)


df['Sentiment'].unique()


df.info()


df.describe()


postcount=df['Sentiment'].value_counts()
plt.bar(x= postcount.index,height=postcount.values,color=['red','blue','pink'])
plt.title('postcount')
plt.xlabel('sentiments')
plt.ylabel('count')
plt.show()


import matplotlib.pyplot as plt

postcount = df['Sentiment'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(postcount.values, labels=postcount.index, autopct='%1.1f%%', colors=['green', 'blue', 'pink'], startangle=140)
plt.title('Sentiment Distribution')
plt.show()



print("Class Distribution:")
print(df["Sentiment"].value_counts())


# Map sentiment labels to numerical values
label_map = {"NU": 0, "N": 1, "P": 2}  # NU: Neutral, N: Negative, P: Positive
df["Sentiment"] = df["Sentiment"].map(label_map)



import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from collections import Counter

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")


# Split dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Comment"], df["Sentiment"], test_size=0.2, random_state=42
)


# Load MiniLM tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLM-L6-H384-uncased")
model = AutoModelForSequenceClassification.from_pretrained("nreimers/MiniLM-L6-H384-uncased", num_labels=3).to(device)



# Tokenization function
def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")



# Tokenize training and testing texts
train_encodings = tokenize_function(train_texts.tolist())
test_encodings = tokenize_function(test_texts.tolist())



# Custom PyTorch Dataset class
class FacebookSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {key: val.to(device) for key, val in encodings.items()}  # Move encodings to GPU
        self.labels = torch.tensor(labels, dtype=torch.long, device=device)      # Move labels to GPU

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# Create datasets
train_dataset = FacebookSentimentDataset(train_encodings, train_labels.tolist())
test_dataset = FacebookSentimentDataset(test_encodings, test_labels.tolist())



# Compute class distribution and weights
class_counts = Counter(train_labels.tolist())
print("Class counts:", class_counts)
total_samples = sum(class_counts.values())
class_weights = {label: total_samples / count for label, count in class_counts.items()}
print("Class Weights:", class_weights)
weights_tensor = torch.tensor([class_weights[i] for i in range(3)], dtype=torch.float).to(device)



# Define custom weighted loss function
loss_function = nn.CrossEntropyLoss(weight=weights_tensor)



# Custom Trainer with weighted loss, updated to handle num_items_in_batch
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_function(logits, labels)
        return (loss, outputs) if return_outputs else loss



# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    dataloader_pin_memory=False  # Disable pinning memory to avoid GPU issues
)


# Define metrics function
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }




# Initialize and train the model
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get predictions on test set
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
true = predictions.label_ids

# Plot the confusion matrix
cm = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neutral", "Negative", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("📊 Confusion Matrix")
plt.show()



trainer.train()


from lime.lime_text import LimeTextExplainer

# Wrapper class for LIME
class_names = ["Neutral", "Negative", "Positive"]

def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

# Create LIME explainer
explainer = LimeTextExplainer(class_names=class_names)



# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")



# Save the model and tokenizer
model.save_pretrained("./fb_sentiment_model")
tokenizer.save_pretrained("./fb_sentiment_model")


# Reverse label map for predictions
reverse_label_map = {v: k for k, v in label_map.items()}



# Prediction function with temperature scaling
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)
    predicted_label = reverse_label_map[prediction.item()]
    print(f"Confidence: {confidence.item():.2f}")
    return predicted_label if confidence.item() > 0.6 else "Uncertain 🤔"




# Test predictions
print(predict("This movie was okay, but not great."))
print(predict("I expected better from this series."))
print(predict("Absolutely terrible, waste of time."))
print(predict("An absolute masterpiece!"))


exp = explainer.explain_instance(
    "I expected better from this series.",
    predict_proba,
    num_features=10,
    labels=[0, 1, 2]
)


exp.show_in_notebook(text=True)