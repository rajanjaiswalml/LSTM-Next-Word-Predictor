import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------
# Page Configuration
# --------------------------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    layout="centered",
)

# --------------------------------------------
# Custom CSS
# --------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"]{
    font-family: 'Segoe UI', sans-serif;
}

.main{
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
}

.title{
    font-size:48px;
    font-weight:700;
    text-align:center;
    color:white;
    margin-bottom:5px;
}

.subtitle{
    text-align:center;
    color:#CBD5E1;
    margin-bottom:40px;
    font-size:18px;
}

.result-box{
    background:#1E293B;
    border-radius:15px;
    padding:20px;
    color:white;
    font-size:24px;
    text-align:center;
    border:1px solid #334155;
    margin-top:20px;
}

.stTextInput>div>div>input{
    border-radius:12px;
    padding:12px;
    font-size:18px;
}

.stButton>button{
    width:100%;
    border-radius:12px;
    height:55px;
    font-size:18px;
    font-weight:bold;
    background:linear-gradient(90deg,#3B82F6,#8B5CF6);
    color:white;
    border:none;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#2563EB,#7C3AED);
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------
# Load Model
# --------------------------------------------
@st.cache_resource
def load_resources():

    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()

# --------------------------------------------
# Generate Text
# --------------------------------------------
def generate_text(seed_text, n_words):

    text = seed_text

    for _ in range(n_words):

        token_list = tokenizer.texts_to_sequences([text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_len-1,
            padding='pre'
        )

        prediction = model.predict(token_list, verbose=0)

        predicted_index = np.argmax(prediction)

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        if output_word == "":
            break

        text += " " + output_word

    return text

# --------------------------------------------
# UI
# --------------------------------------------
st.markdown('<div class="title">🤖 Next Word Predictor</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Generate intelligent text using your LSTM model.</div>',
    unsafe_allow_html=True
)

sentence = st.text_input(
    "Enter a sentence",
    placeholder="Example: Artificial Intelligence is"
)

num_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=20,
    value=5
)

if st.button("✨ Generate Text"):

    if sentence.strip() == "":
        st.warning("Please enter a sentence.")
    else:

        with st.spinner("Generating..."):

            result = generate_text(sentence, num_words)

        st.markdown(
            f'<div class="result-box">{result}</div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

st.caption("Built with ❤️ using TensorFlow, LSTM & Streamlit")