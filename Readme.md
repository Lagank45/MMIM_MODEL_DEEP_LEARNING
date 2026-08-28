# Enhanced MMIM for Multimodal Sentiment Analysis

An enhanced implementation of **Multimodal Mutual Information Maximization (MMIM)** for multimodal sentiment analysis using **text, audio, and visual modalities**.

## Overview

This project extends the original MMIM architecture with improved multimodal interaction, adaptive fusion, and robustness mechanisms.

### Key Improvements

- **GCN Fusion:** Treats text, audio, and visual embeddings as graph nodes and learns cross-modal interactions through message passing with residual connections.
- **Adaptive Gated Fusion:** Dynamically learns the importance of each modality and the graph representation during fusion.
- **Modality Dropout:** Randomly drops modalities during training to improve robustness to missing or noisy inputs.
- **Layer Normalization & Dropout:** Improves training stability and generalization.
- **Token Attention Pooling:** Learns to assign higher importance to informative textual tokens.

## Architecture

```text
Text ──→ Text Encoder ──→ Token Pooling ──┐
                                          │
Audio ─→ RNN Encoder ─────────────────────┤
                                          ├──→ Projection
Video ─→ RNN Encoder ─────────────────────┘
                                               │
                                               ↓
                                        Modality Dropout
                                               │
                                               ↓
                                          GCN Fusion
                                               │
                                               ↓
                                    Graph Embedding (g)
                                               │
                                               ↓
                                    Adaptive Gated Fusion
                                               │
                                               ↓
                                         Hybrid Fusion
                                               │
                                               ↓
                                      LayerNorm + Dropout
                                               │
                                               ↓
                                          Classifier
