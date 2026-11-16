# 🧠 NeuralType · Next Word Prediction with PyTorch LSTM

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Architecture-Stacked%20LSTM-blueviolet?style=for-the-badge" alt="Architecture" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

An end-to-end Natural Language Processing (NLP) deep learning system that predicts and generates subsequent tokens in a sentence using a stacked **Long Short-Term Memory (LSTM)** neural network. Features a dark-themed, glassmorphic interactive web dashboard built with **Streamlit** for real-time text completion, confidence scoring, and sampling exploration.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Architecture & Specifications](#-architecture--specifications)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Running the Streamlit Application](#-running-the-streamlit-application)
- [Model Training](#-model-training)
- [How It Works](#-how-it-works)
- [Hyperparameters & Metrics](#-hyperparameters--metrics)
- [License](#-license)

---

## 🔍 Overview

Language modeling at the word level involves estimating the conditional probability distribution of the next word given a sequence of preceding words:

$$P(w_t \mid w_{t-1}, w_{t-2}, \dots, w_{t-k})$$

This project implements a sequence-to-word model trained on literary text (Shakespearean corpus) capable of auto-completing prompts word-by-word with controllable creativity (temperature) and diversity (Top-$K$ filtering).

---

## 🏗️ Architecture & Specifications

The network employs a deep recurrent architecture optimized for sequential context retention:

```
Input Tokens [x_1, x_2, ..., x_t]
             │
             ▼
     ┌───────────────┐
     │ Embedding (E) │  (vocab_size × 256)
     └───────┬───────┘
             │
             ▼
     ┌───────────────┐
     │  LSTM Layer 1 │  (hidden_dim: 512, batch_first=True)
     └───────┬───────┘
             │  (Dropout = 0.3)
             ▼
     ┌───────────────┐
     │  LSTM Layer 2 │  (hidden_dim: 512, batch_first=True)
     └───────┬───────┘
             │
             ▼
     ┌───────────────┐
     │ Linear / Dense│  (512 ➔ vocab_size)
     └───────┬───────┘
             │
             ▼
   Next Token Distribution (Softmax + Temperature)
```

| Component | Value / Description |
|---|---|
| **Model Type** | Stacked LSTM (`NextWordLSTM`) |
| **Vocabulary Size** | ~3,233 unique words |
| **Embedding Dimension** | 256 |
| **Hidden Units** | 512 per layer |
| **LSTM Layers** | 2 |
| **Dropout** | 0.3 |
| **Context Window ($k$)** | 30 words (`seq_len = 30`) |
| **Optimizer** | AdamW ($\text{lr} = 0.002$) |
| **Loss Function** | Categorical Cross-Entropy (`nn.CrossEntropyLoss`) |

---

## ✨ Key Features

- **Real-Time Next Word Inference:** Instant multi-word recursive generation based on user prompt.
- **Controllable Sampling Dynamics:**
  - **Temperature Scaling ($T$):** Flatten or sharpen logits to balance coherence vs. hallucination/creativity.
  - **Top-$K$ Candidate Filtering:** Restrict candidate tokens to top $K$ probabilities before sampling.
- **Top Candidates Probability Visualizer:** Inspect predictive confidence and probability distribution across candidate tokens.
- **Dark Glassmorphism Interface:** Custom CSS UI designed for modern developer and AI dashboards.
- **Standalone Checkpoint:** Self-contained `.pth` weights file storing model parameters, vocabulary index mappings, and hyperparameters.

---

## 📁 Repository Structure

```plaintext
Next_Word_Prediction_LSTM/
├── app.py                      # Interactive Streamlit application
├── LSTM_Next_word_pred.ipynb   # Complete training, preprocessing & evaluation notebook
├── next_word_lstm_full.pth     # Pretrained checkpoint (weights + vocabulary + metadata)
├── text.txt                    # Training corpus (Tiny Shakespeare dataset)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR-USERNAME>/Next_Word_Prediction_LSTM.git
cd Next_Word_Prediction_LSTM
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Streamlit Application

Launch the local web server:

```bash
streamlit run app.py
```

The application will launch automatically in your default browser at `http://localhost:8501`.

### App Controls:
- **Input Text:** Enter any starting phrase or prompt.
- **Words to Predict:** Choose how many successive tokens to generate (e.g. 5–30 words).
- **Creativity (Temperature):**
  - Lower ($T < 0.7$): More conservative, deterministic predictions.
  - Higher ($T > 1.0$): Diverse and creative, novel word combinations.
- **Top-$K$ Filter:** Number of highest-probability tokens to retain before sampling.

---

## 🏋️ Model Training

To retrain the model on custom text:

1. Place your raw text dataset into `text.txt`.
2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook LSTM_Next_word_pred.ipynb
   ```
3. The training pipeline will:
   - Clean, tokenize, and construct the vocabulary ($O(\text{freq})$ filtering).
   - Generate sliding sequences with length `seq_len = 30`.
   - Train the 2-layer LSTM model with AdamW over 10 epochs.
   - Export `next_word_lstm_full.pth` containing model weights and word-to-index dictionaries.

---

## 🔬 How It Works

### 1. Token Preprocessing & Context Slicing
Incoming text is tokenized into word units, stripped of irregular symbols, and converted to integer tokens:
$$x = [\text{idx}(w_{t-29}), \dots, \text{idx}(w_t)]$$

### 2. Recurrent Context Processing
The sequence passes through embedding vectors followed by gated recurrent units:
$$i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})$$
$$f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})$$
$$g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})$$
$$o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})$$
$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$
$$h_t = o_t \odot \tanh(c_t)$$

### 3. Temperature-Scaled Softmax
Before sampling, the output logits $z$ from the final linear projection are adjusted via temperature $T$:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

---

## 📊 Hyperparameters & Metrics

| Parameter | Value |
|---|---|
| **Batch Size** | 128 |
| **Sequence Length** | 30 |
| **Embedding Size** | 256 |
| **LSTM Hidden Units** | 512 |
| **Number of Layers** | 2 |
| **Dropout** | 0.3 |
| **Learning Rate** | 0.002 (AdamW) |
| **Training Epochs** | 10 |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
