import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import traceback  # ✅ Import traceback module

# ✅ Get Current Date & Time
current_time = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")  # 12-hour format with AM/PM

# ✅ Apply Custom CSS with Enhanced Styling
st.markdown(
    """
    <style>
        /* General Styles */
        .stApp {
            background-color: #f0f2f6; /* Light grey background */
        }
        .main {background-color: #f0f2f6; color: black;}
        h1 {color: #2E86C1; text-align: center;}
        
        /* Title and Subtitle Styles */
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #2E86C1; /* Blue text */
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            text-align: center;
            color: #555; /* Medium grey text */
        }

        /* Footer Styles */
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 20px;
            color: #777; /* Lighter grey */
        }

        /* Text Area Styles */
        div[data-testid="stTextArea"] label {
            color: black !important; /* Make 'Enter your text' label visible */
            font-weight: bold;
            font-size: 16px;
        }
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: black !important;
            font-size: 16px;
            border-radius: 10px;
            caret-color: black !important; /* Ensures cursor is visible */
            outline: none !important;
        }
        .stTextArea textarea::placeholder {
            color: #555555 !important; /* Darker placeholder text */
            font-weight: bold;
        }
        .stTextArea textarea:focus {
            border: 2px solid #2E86C1 !important;
            caret-color: black !important;
        }

        /* Button Styles */
        .stButton>button {
            background-color: #2E86C1; /* Blue button */
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1B4F72; /* Darker blue on hover */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Load Fine-Tuned MiniLM Model & Tokenizer from Local Folder
@st.cache_resource
def load_model():
    model_path = "./fb_sentiment_model"  # Path to your fine-tuned MiniLM model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(traceback.format_exc())  # ✅ Use traceback to print detailed error
        st.stop()

model, tokenizer, device = load_model()

# ✅ Prediction Function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return {0: "😐 Neutral", 1: "😡 Negative", 2: "😊 Positive"}[prediction]

# ✅ Streamlit UI
st.markdown('<p class="title">💬 Facebook Sentiment Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter a comment below to analyze its sentiment using your fine-tuned MiniLM model.</p>', unsafe_allow_html=True)

# ✅ Input Text Area with Visible Placeholder & Cursor
user_input = st.text_area("✍️ Enter your post here:", placeholder="Type your Facebook post here...", height=150)

if st.button("🔍 Predict Sentiment"):
    if user_input.strip():
        try:
            sentiment = predict_sentiment(user_input)
            st.markdown(f"""
            <div style="background-color:#DFF0D8; padding:10px; border-radius:10px;">
                <p style="color:#155724; font-size:18px; font-weight:bold; text-align:center;">
                    Predicted Sentiment: {sentiment}
                </p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.error(traceback.format_exc())  # ✅ Use traceback to print detailed error
    else:
        st.markdown("""
        <div style="background-color:#FFF3CD; padding:10px; border-radius:10px;">
            <p style="color:#856404; font-size:18px; font-weight:bold; text-align:center;">
                ⚠️ Please enter some text to analyze.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ✅ Footer
st.markdown('<p class="footer">🔧 Built with Streamlit & Hugging Face Transformers 🚀</p>', unsafe_allow_html=True)
st.markdown('<p class="footer">👤 Developed by <b>K. AJITH</b></p>', unsafe_allow_html=True)
st.markdown(f'<p class="footer">📅 Current Date & Time: <b>{current_time}</b></p>', unsafe_allow_html=True)