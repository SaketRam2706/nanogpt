# Small-Scale GPT

A minimal implementation of a GPT (Generative Pre-trained Transformer) model trained on Shakespeare's text. This project demonstrates the core concepts of transformer-based language models in a simple and educational way.

## Overview

This implementation includes:
- A small scale GPT language model
- Character-level tokenization
- Training on Shakespeare's text
- Text generation capabilities

## Features

- Small scale GPT model implementation
- Character-level tokenization (vocabulary size of 65 characters)
- Training on Shakespeare's text dataset
- Text generation with configurable sequence length
- PyTorch-based implementation
- AdamW optimizer for training

## Implementation Details

### Model Architecture
- Token embedding layer
- Simple bigram prediction
- Cross-entropy loss for training

### Data Processing
- Character-level tokenization
- Train/validation split (90/10)
- Batch processing with configurable context length
- Sliding window approach for sequence generation

## Usage

### Training
```python
# Initialize model
model = BigramLanguageModel(vocab_size)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for steps in range(100):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

### Text Generation
```python
# Generate new text
context = torch.zeros((1, 1), dtype=torch.long)
generated_text = model.generate(context, max_new_tokens=500)
print(decode(generated_text[0].tolist()))
```

## Requirements

- Python 3.6+
- PyTorch
- Shakespeare text dataset (included in the repository)

## Project Structure

- `tiny_shake.ipynb`: Main implementation notebook
- `input.txt`: Shakespeare text dataset


