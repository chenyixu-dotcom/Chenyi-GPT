# Vanilla Transformer

**Version:** 0.1  
**Framework:** PyTorch  
**Status:** Experimental

A lightweight, decoder-only Transformer built from scratch in PyTorch to explore the fundamentals of language modeling, attention mechanisms, and autoregressive text generation.

## Overview

Vanilla Transformer is an experimental language model trained to predict the next token in a sequence. It generates text autoregressively, using previously generated tokens as context.

The project focuses on understanding Transformer architecture, training, and inference without relying on pretrained language models.

## Features

- Custom Transformer model implemented in PyTorch
- Autoregressive next-token prediction
- Vocabulary-based tokenization
- Temperature-controlled text generation
- Top-k sampling
- Sentence completion and next-sentence generation experiments
- Model checkpoint saving and loading
- Interactive terminal interface

## How It Works

**Training**

1. Tokenize the training text and map tokens to vocabulary IDs.
2. Feed token sequences into the Transformer.
3. Predict the next token at each position.
4. Calculate the prediction loss.
5. Update model parameters through backpropagation and gradient descent.
6. Save the trained model weights.

**Inference**

1. Convert the user's input into token IDs.
2. Load the trained Transformer weights.
3. Predict a probability distribution over the next token.
4. Sample a token using temperature and top-k sampling.
5. Append the generated token to the sequence.
6. Repeat until a stopping condition is reached.

## Limitations

Version 0.1 is an educational prototype rather than a production language model.

- Limited training data and vocabulary
- Short, repetitive, or incoherent generations
- Limited ability to maintain long-range context
- Sentence boundaries are not always handled reliably
- No instruction tuning or preference optimization

## Future Improvements

- [ ] Improve training data quality and diversity
- [ ] Implement more robust sentence-boundary detection
- [ ] Experiment with top-p sampling and repetition penalties
- [ ] Improve tokenization
- [ ] Evaluate model performance on held-out text
- [ ] Experiment with larger Transformer architectures

## Version History

**v0.1 — Initial Prototype**

- Implemented a custom Transformer language model
- Added training and checkpoint loading
- Implemented autoregressive text generation
- Added temperature and top-k sampling
- Built an interactive terminal interface

## Purpose

This project is intended as a hands-on exploration of how Transformer-based language models learn and generate text, from model architecture and optimization to token sampling and inference.