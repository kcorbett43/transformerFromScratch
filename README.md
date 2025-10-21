## Overview

This project aims to implement the transformer and attention network described in the "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762) paper. The attention network and transformer architecture led to the vast advancements in machine learning that we see today and this project will dive into the fundamentals of these breakthroughs.

- `transformer.ipynb` – the main notebook where the model is built step-by-step.

### Current Implementation:

### **1. Data & Tokenization**
- Uses the **GPT-2 tokenizer** (`GPT2Tokenizer`) with padding token set to `eos_token`.  
- Streams text from the **Wikimedia 2023-11 English** dataset using the `datasets` library.  
- Tokenizes and pads to a fixed `max_length`, producing `input_ids` and `attention_mask`.

### **2. Model Components**
- **PositionalEncoding:** Implements sinusoidal encodings and adds them to token embeddings.  
- **Attention:** Defines a single-head attention layer.  
- **MultiHeadAttention:** Implements learned projections for Q, K, V, and O; performs scaled dot-product attention.  
- **FeedForward:** Standard two-layer MLP with ReLU activation and dropout.  
- **EncoderTransformer:** Adds multi-head attention and feed-forward layers with residuals and normalization.  
- **DecoderTransformer:** Implements masked self-attention and feed-forward sublayers with layer norms.  
- **GPT2:** A decoder-only stack that combines embeddings, positional encodings, several decoder blocks, and a final linear projection layer for logits.

### **3. Training Setup**
- Defines hyperparameters (vocab size, embedding dimension, number of heads/layers, feed-forward width, etc.).  
- Streams mini-batches from Wikipedia and performs next-token prediction using cross-entropy loss.  
- Optimizes with Adam; logs loss every 10 batches.  
- Saves model checkpoints (`.pth`) locally and to Google Drive.

### **4. Text Generation**
- Implements top-k sampling with temperature for autoregressive text generation.  
- Iteratively appends predicted tokens until reaching `max_length` or an EOS token.

---
