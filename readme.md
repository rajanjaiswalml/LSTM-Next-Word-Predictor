# Next Word Predictor using LSTM

A Deep Learning based Next Word Prediction web application built using TensorFlow, Keras, LSTM and Streamlit.

## Project Overview

This project predicts the next most probable word from a given text sequence using an LSTM (Long Short-Term Memory) neural network.

The objective of this project was to understand sequence modeling, text preprocessing, tokenization, and deployment of deep learning models through an interactive web application.

---

## Features

- Next Word Prediction
- Interactive Streamlit Interface
- LSTM-based Deep Learning Model
- Text Tokenization
- Sequence Padding
- Real-time Prediction

---

## Model Performance

| Metric | Value |
|---------|---------|
| Dataset Size | 10,000 Text Samples |
| Epochs | 100 |
| Training Accuracy | 72.93% |
| Training Loss | 1.2153 |

---

## Tech Stack

- Python
- TensorFlow
- Keras
- LSTM
- NumPy
- Streamlit

---

## Project Structure

```
LSTM-Next-Word-Predictor
│
├── app.py
├── lstm_model.h5
├── tokenizer.pkl
├── max_len.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── assets
```

---

## Installation

```bash
git clone https://github.com/rajanjaiswalml/LSTM-Next-Word-Predictor.git

cd LSTM-Next-Word-Predictor

pip install -r requirements.txt

streamlit run app.py
```

---

## Future Improvements

- Larger Dataset
- Better Hyperparameter Tuning
- Transformer Architecture
- Beam Search
- Better Prediction Accuracy

---

## Author

Rajan Jaiswal

Machine Learning Enthusiast
