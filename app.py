import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import time
import pandas as pd

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralType · Next Word AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS  — dark glassmorphism premium design
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b3e 40%, #0a0a1a 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 40, 0.95);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

/* ── Slider ── */
.stSlider > div > div > div > div { background: #6366f1 !important; }
.stSlider > div > div > div { background: rgba(99,102,241,0.2) !important; }

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6);
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
}
.stButton > button:active { transform: translateY(0); }

/* ── Text areas ── */
.stTextArea textarea {
    background: rgba(15, 20, 50, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.05rem !important;
    resize: none !important;
    transition: border-color 0.3s ease;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* ── Labels ── */
.stTextArea label, .stSlider label {
    color: #a5b4fc !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
}

/* ── Custom card component ── */
.glass-card {
    background: rgba(15, 20, 55, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc 0%, #c4b5fd 50%, #f9a8d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}

.hero-subtitle {
    color: #64748b;
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 1.5rem;
}

.stat-pill {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #a5b4fc;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    letter-spacing: 0.3px;
}

.predicted-word {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.25));
    border: 1px solid rgba(139,92,246,0.5);
    border-radius: 8px;
    padding: 2px 8px;
    margin: 2px 3px;
    color: #c4b5fd;
    font-weight: 600;
    font-size: 1.05rem;
    animation: fadein 0.3s ease;
}

.seed-word {
    display: inline-block;
    color: #e2e8f0;
    font-size: 1.05rem;
    margin: 2px 3px;
    font-weight: 400;
}

@keyframes fadein {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}

.output-box {
    background: rgba(10, 12, 35, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    min-height: 80px;
    line-height: 2.2rem;
    word-break: break-word;
}

.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.5rem;
}

.sidebar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #a5b4fc;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}

.temp-low  { color: #34d399; font-weight: 600; }
.temp-mid  { color: #fbbf24; font-weight: 600; }
.temp-high { color: #f87171; font-weight: 600; }

.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-green  { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-purple { background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }

.divider {
    border: none;
    border-top: 1px solid rgba(99,102,241,0.15);
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  MODEL DEFINITION  (must match training notebook exactly)
# ─────────────────────────────────────────────────────────────
class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        out = self.dropout(out)
        logits = self.fc(out[:, -1, :])
        return logits, hidden


# ─────────────────────────────────────────────────────────────
#  LOAD CHECKPOINT  (cached — only runs once)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_vocab(path: str = "next_word_lstm_full.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)

    word_to_idx = ckpt["word_to_idx"]
    idx_to_word = ckpt["idx_to_word"]
    seq_len     = ckpt["seq_len"]
    vocab_size  = len(word_to_idx)
    embed_dim   = ckpt["embed_dim"]
    hidden_dim  = ckpt["hidden_dim"]
    num_layers  = ckpt["num_layers"]

    model = NextWordLSTM(vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    return model, word_to_idx, idx_to_word, seq_len, device, vocab_size, embed_dim, hidden_dim, num_layers


# ─────────────────────────────────────────────────────────────
#  GENERATION HELPER
# ─────────────────────────────────────────────────────────────
def predict_next_words(model, word_to_idx, idx_to_word, device,
                       seed_text: str, num_words: int, seq_len: int,
                       temperature: float, top_k: int = 5):
    """
    Returns:
      generated_words  : list of all words (seed + predicted)
      seed_length      : how many words came from the seed
      topk_data        : list of dicts {word, prob} for the LAST prediction step
    """
    words = re.findall(r'\b\w+\b|[.,!?;]', seed_text.lower())
    seed_length = len(words)
    topk_data = []

    for step in range(num_words):
        context = words[-seq_len:] if len(words) >= seq_len else words
        indices = [word_to_idx.get(w, 1) for w in context]   # 1 = <UNK>
        if not indices:
            break
        x = torch.tensor([indices], dtype=torch.long).to(device)

        with torch.no_grad():
            logits, _ = model(x)
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1).squeeze()

        # ✅ FIX: Zero out special tokens so they are NEVER sampled
        # Index 0 = <PAD>, Index 1 = <UNK> — mask both out
        probs[0] = 0.0  # <PAD>
        probs[1] = 0.0  # <UNK>
        # Re-normalise so probabilities still sum to 1
        probs = probs / probs.sum()

        # Collect top-k for the last step (also filter out special tokens)
        if step == num_words - 1:
            top_probs, top_ids = torch.topk(probs, k=min(top_k + 2, probs.size(0)))
            topk_data = [
                {"word": idx_to_word.get(int(i), ""), "probability": float(p) * 100}
                for i, p in zip(top_ids, top_probs)
                if idx_to_word.get(int(i), "") not in ("", "<UNK>", "<PAD>")
            ][:top_k]  # Keep only requested top-k after filtering

        pred_idx = torch.multinomial(probs, 1).item()
        predicted_word = idx_to_word.get(pred_idx, "")
        if predicted_word and predicted_word not in ("<UNK>", "<PAD>"):
            words.append(predicted_word)

    return words, seed_length, topk_data


def render_output_html(words, seed_length):
    """Colour-coded output: seed words normal, predicted words highlighted."""
    parts = []
    for i, w in enumerate(words):
        if i < seed_length:
            parts.append(f'<span class="seed-word">{w}</span>')
        else:
            parts.append(f'<span class="predicted-word">{w}</span>')
    return " ".join(parts)


def clean_text(words):
    result = " ".join(words)
    result = re.sub(r'\s+([.,!?;])', r'\1', result)
    return result


# ─────────────────────────────────────────────────────────────
#  LOAD MODEL  (with graceful error handling)
# ─────────────────────────────────────────────────────────────
try:
    (model, word_to_idx, idx_to_word,
     seq_len, device, vocab_size,
     embed_dim, hidden_dim, num_layers) = load_model_and_vocab()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 NeuralType</div>', unsafe_allow_html=True)

    if model_loaded:
        device_str = "CUDA GPU 🚀" if str(device) == "cuda" else "CPU 💻"
        st.markdown(f"""
        <div class="glass-card" style="padding:1rem;">
            <div class="section-label">Model Info</div>
            <div style="color:#e2e8f0; font-size:0.85rem; line-height:2;">
                <span class="badge badge-green">Loaded</span><br>
                📚 <b>Vocab:</b> {vocab_size:,} words<br>
                🔢 <b>Embed Dim:</b> {embed_dim}<br>
                🏛️ <b>Hidden Dim:</b> {hidden_dim}<br>
                📐 <b>LSTM Layers:</b> {num_layers}<br>
                📏 <b>Context (seq):</b> {seq_len} words<br>
                ⚙️ <b>Device:</b> {device_str}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Model not loaded")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Generation Settings</div>', unsafe_allow_html=True)

    num_words = st.slider("Words to Predict", min_value=1, max_value=15, value=10,
                          help="LSTM accuracy is best at 10–15 words. Beyond that, output becomes less coherent.")

    temperature = st.slider("Temperature (Creativity)", min_value=0.1, max_value=2.0,
                            value=0.7, step=0.05,
                            help="Low = safe & repetitive | High = creative & wild")

    if temperature < 0.5:
        temp_label = '<span class="temp-low">❄️ Very Deterministic</span>'
    elif temperature < 1.0:
        temp_label = '<span class="temp-mid">⚖️ Balanced</span>'
    elif temperature < 1.5:
        temp_label = '<span class="temp-high">🔥 Creative</span>'
    else:
        temp_label = '<span class="temp-high" style="color:#f87171;">🌋 Very Wild</span>'
    st.markdown(temp_label, unsafe_allow_html=True)

    top_k_display = st.slider("Top-K Predictions to Show", min_value=3, max_value=15, value=7)

    show_animated = st.toggle("Word-by-word animation", value=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#475569; font-size:0.75rem; line-height:1.7;">
        <b style="color:#6366f1;">How it works:</b><br>
        1. Type any starting phrase<br>
        2. The LSTM reads the last 30 words<br>
        3. Predicts probabilities over 3,233 words<br>
        4. Samples from the distribution<br>
        5. Repeats until target word count
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="glass-card">
    <div class="hero-title">NeuralType ✍️</div>
    <div class="hero-subtitle">LSTM-powered Next Word Prediction • Type a phrase and watch the AI continue it</div>
    <span class="stat-pill">📖 Shakespeare Corpus</span>
    <span class="stat-pill">🔁 2-Layer LSTM</span>
    <span class="stat-pill">🎯 3,233 Vocab</span>
    <span class="stat-pill">⚡ PyTorch</span>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"❌ Could not load model checkpoint. Make sure **next_word_lstm_full.pth** is in the same folder.\n\n`{load_error}`")
    st.stop()

# ── Initialise seed value in session state (only once) ──
if "seed_value" not in st.session_state:
    st.session_state["seed_value"] = "to be or not to be"

# ── Input section ──
col_input, col_examples = st.columns([3, 1])

with col_examples:
    st.markdown('<div class="section-label">Quick Examples</div>', unsafe_allow_html=True)
    examples = [
        "the king of",
        "once upon a time",
        "all the world is",
        "love is blind",
        "to be or not to be",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            # ✅ FIX: update our OWN state variable, NOT the widget key
            st.session_state["seed_value"] = ex
            st.rerun()

with col_input:
    st.markdown('<div class="section-label">Your Seed Text</div>', unsafe_allow_html=True)
    # ✅ FIX: widget uses `value=` from our state, NOT `key=` bound to state
    seed_text = st.text_area(
        label="seed",
        label_visibility="collapsed",
        placeholder="Start typing anything… e.g. 'to be or not to be'",
        value=st.session_state["seed_value"],
        height=110,
    )

st.markdown("")

# ── Generate button ──
col_btn, col_clear = st.columns([3, 1])
with col_btn:
    generate_clicked = st.button("🚀  Generate Prediction", type="primary")
with col_clear:
    clear_clicked = st.button("🗑️  Clear Output")

if clear_clicked:
    if "output_words" in st.session_state:
        del st.session_state["output_words"]
        del st.session_state["output_seed_len"]
        del st.session_state["output_topk"]
    st.rerun()

# ── Run generation ──
if generate_clicked:
    if not seed_text.strip():
        st.warning("⚠️ Please enter some seed text first!")
    else:
        with st.spinner("Thinking…"):
            words, seed_length, topk_data = predict_next_words(
                model, word_to_idx, idx_to_word, device,
                seed_text, num_words, seq_len, temperature, top_k_display
            )
        st.session_state["output_words"]    = words
        st.session_state["output_seed_len"] = seed_length
        st.session_state["output_topk"]     = topk_data

# ── Display results ──
if "output_words" in st.session_state:
    words       = st.session_state["output_words"]
    seed_length = st.session_state["output_seed_len"]
    topk_data   = st.session_state["output_topk"]

    predicted_count = len(words) - seed_length

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Result header
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding:1rem;">
            <div style="font-size:1.8rem; font-weight:700; color:#a5b4fc;">{seed_length}</div>
            <div style="color:#64748b; font-size:0.8rem;">Seed Words</div>
        </div>""", unsafe_allow_html=True)
    with col_r2:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding:1rem;">
            <div style="font-size:1.8rem; font-weight:700; color:#c4b5fd;">{predicted_count}</div>
            <div style="color:#64748b; font-size:0.8rem;">Words Predicted</div>
        </div>""", unsafe_allow_html=True)
    with col_r3:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding:1rem;">
            <div style="font-size:1.8rem; font-weight:700; color:#f9a8d4;">{temperature:.2f}</div>
            <div style="color:#64748b; font-size:0.8rem;">Temperature</div>
        </div>""", unsafe_allow_html=True)

    # ── Animated word-by-word output ──
    st.markdown('<div class="section-label" style="margin-top:1rem;">Generated Output</div>', unsafe_allow_html=True)

    output_placeholder = st.empty()

    if show_animated and generate_clicked:
        # Reveal predicted words one by one
        shown = list(words[:seed_length])
        for i in range(seed_length, len(words)):
            shown.append(words[i])
            html = render_output_html(shown, seed_length)
            output_placeholder.markdown(
                f'<div class="output-box">{html}</div>',
                unsafe_allow_html=True
            )
            time.sleep(0.10)
    else:
        html = render_output_html(words, seed_length)
        output_placeholder.markdown(
            f'<div class="output-box">{html}</div>',
            unsafe_allow_html=True
        )

    # ── Clean text copy box ──
    clean = clean_text(words)
    st.markdown('<div class="section-label" style="margin-top:1rem;">📋 Clean Text (copy me)</div>', unsafe_allow_html=True)
    st.code(clean, language=None)

    # ── Top-K probability chart ──
    if topk_data:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🎯 Top Candidate Words at Last Step</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#64748b; font-size:0.8rem; margin-bottom:0.75rem;">Probability distribution over the final predicted word — higher bars = more likely.</div>', unsafe_allow_html=True)

        df = pd.DataFrame(topk_data).sort_values("probability", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df["probability"] = df["probability"].round(2)
        df.columns = ["Word", "Probability (%)"]

        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            st.bar_chart(df.set_index("Word")["Probability (%)"], color="#6366f1", height=260)
        with col_table:
            st.dataframe(
                df,
                width="stretch",
                height=260,
                column_config={
                    "Word": st.column_config.TextColumn("🔤 Word"),
                    "Probability (%)": st.column_config.ProgressColumn(
                        "📊 Probability",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    )
                }
            )

    # ── Legend ──
    st.markdown("""
    <div style="margin-top:0.5rem; color:#475569; font-size:0.78rem;">
        <span style="background:rgba(99,102,241,0.25); border:1px solid rgba(139,92,246,0.5);
              border-radius:6px; padding:1px 7px; color:#c4b5fd; font-weight:600; margin-right:6px;">highlighted</span>
        = AI-generated words &nbsp;|&nbsp;
        <span style="color:#e2e8f0; margin-right:6px;">normal</span>
        = your seed text
    </div>
    """, unsafe_allow_html=True)
