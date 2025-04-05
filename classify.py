import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm

# ✅ Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Load the 3-class sentiment model + tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()  # Set to eval mode

# ✅ Load dataset
df = pd.read_csv("netflix_cleaned.csv")
df["Comment"] = df["Comment"].fillna("").astype(str)  # Ensure no NaNs

# ✅ Process text in BATCHES
batch_size = 512  # Adjust based on GPU memory
comments = df["Comment"].tolist()
sentiments = []

# 🚀 Run inference in batches for max speed
for i in tqdm(range(0, len(comments), batch_size), desc="Processing", unit="batch"):
    batch = comments[i:i + batch_size]

    # ✅ Tokenize efficiently with explicit truncation & padding
    inputs = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)

    # ✅ Model forward pass (optimized with `torch.no_grad()` and `float16`)
    with torch.no_grad():
        outputs = model(**inputs).logits.to(torch.float16)  # Convert to float16 for speed
    
    # ✅ Convert logits to probabilities & get predictions
    predictions = torch.argmax(F.softmax(outputs, dim=-1), dim=-1).cpu().numpy()

    # ✅ Convert labels to P, NU, N
    sentiment_map = {0: "N", 1: "NU", 2: "P"}
    sentiments.extend([sentiment_map[p] for p in predictions])

# ✅ Map results back to DataFrame
df["Sentiment"] = sentiments

# ✅ Save labeled dataset
df.to_csv("netflix_labeled.csv", index=False)
print("🚀 Classification completed! Saved as 'netflix_labeled.csv' ✅")
