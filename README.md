---
title: Sentiment Review App Ab
emoji: 🏢
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.34.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🎬 IMDB Movie Review Sentiment Analyzer

A simple yet effective sentiment analysis app that predicts whether a movie review is positive or negative using a trained Simple RNN model.

## 🚀 Features

- **Real-time sentiment prediction** - Get instant results as you type
- **Confidence scoring** - See how confident the model is about its prediction
- **Simple RNN architecture** - Lightweight model trained on IMDB dataset
- **User-friendly interface** - Clean Gradio interface with examples

## 🤖 Model Details

- **Architecture**: Simple RNN with Embedding layer
- **Training Data**: IMDB Movie Reviews Dataset (50,000 reviews)
- **Input**: Text reviews (up to 500 words)
- **Output**: Sentiment (Positive/Negative) + Confidence Score (0-1)

## 📊 How it Works

1. **Text Preprocessing**: Your review is tokenized and converted to numerical sequences
2. **Padding**: Sequences are padded to uniform length (500 tokens)
3. **Prediction**: The RNN model processes the sequence and outputs a probability
4. **Classification**: Scores > 0.5 = Positive 😊, Scores ≤ 0.5 = Negative 😞

## 🎯 Usage

Simply enter your movie review in the text box and click "Submit" to get:
- Sentiment prediction (Positive/Negative)
- Confidence score (0.0 to 1.0)

## 📝 Examples

Try these sample reviews:
- *"This movie was fantastic! The acting was great and the plot was thrilling."* → **Positive**
- *"I was really disappointed with this film. The story was boring and predictable."* → **Negative**

## ⚙️ Technical Stack

- **Framework**: TensorFlow/Keras
- **Interface**: Gradio
- **Deployment**: Hugging Face Spaces
- **Language**: Python

## 🔧 Local Development

To run locally:

```bash
pip install -r requirements.txt
python app.py
```

## 📄 License

This project is open source and available under the MIT License.