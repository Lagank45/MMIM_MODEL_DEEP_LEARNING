# Enhanced MMIM for Multimodal Sentiment Analysis

An enhanced implementation of **Multimodal Mutual Information Maximization (MMIM)** for multimodal sentiment analysis using **Text, Audio, and Visual** modalities.

## Overview

This project extends the original MMIM architecture with improved multimodal interaction, adaptive fusion, and robustness.

### Key Improvements

- **GCN Fusion:** Treats text, audio, and visual embeddings as graph nodes and learns cross-modal interactions through message passing with residual connections.
- **Adaptive Gated Fusion:** Dynamically learns the importance of each modality and the graph representation.
- **Modality Dropout:** Randomly drops modalities during training to improve robustness to missing or noisy inputs.
- **Layer Normalization & Dropout:** Improves training stability and generalization.
- **Token Attention Pooling:** Learns to focus on informative textual tokens.

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
```

## Datasets

The model is designed for multimodal sentiment analysis and can be evaluated on **CMU-MOSI** and **CMU-MOSEI**.

### CMU-MOSI

**CMU-MOSI (Multimodal Opinion-level Sentiment Intensity)** is a multimodal sentiment analysis dataset containing short opinion videos collected from YouTube.

- **2,199 video segments**
- **93 speakers/videos**
- Sentiment scores ranging from **-3 to +3**
- Three modalities:
  - **Text:** Transcribed speech
  - **Audio:** Acoustic and prosodic features
  - **Visual:** Facial and visual features

Sentiment scale:

```text
-3 → Strongly Negative
 0 → Neutral
+3 → Strongly Positive
```

### CMU-MOSEI

**CMU-MOSEI (Multimodal Language Analysis in the Wild)** is a larger and more diverse multimodal sentiment and emotion dataset.

- **23,500+ annotated video segments**
- **1,000+ speakers**
- Videos collected from YouTube
- Sentiment scores ranging from **-3 to +3**
- Three primary modalities:
  - **Text**
  - **Audio**
  - **Visual**

Compared with MOSI, MOSEI provides a larger and more diverse collection of speakers and utterances, making it useful for evaluating model scalability and generalization.

## How the Datasets Are Used

For both MOSI and MOSEI, the three modalities are processed separately before being fused.

```text
                Dataset Sample
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
        Text       Audio      Video
          ↓          ↓          ↓
       Encoder    Encoder    Encoder
          │          │          │
          └──────────┼──────────┘
                     ↓
              Common Projection
                     ↓
             Modality Dropout
                     ↓
                 GCN Fusion
                     ↓
           Graph Embedding (g)
                     ↓
          Adaptive Gated Fusion
                     ↓
              Hybrid Fusion
                     ↓
                Classifier
                     ↓
             Sentiment Score
```

### Training Process

1. Text, audio, and visual features are extracted from each utterance.
2. Each modality is processed using its corresponding encoder.
3. The representations are projected into a common feature space.
4. **Modality dropout** randomly removes modalities during training to improve robustness.
5. Text, audio, and visual embeddings are treated as graph nodes.
6. **GCN message passing** learns relationships between the modalities.
7. The updated graph nodes are pooled to produce a global graph embedding `g`.
8. **Adaptive gated fusion** dynamically weights the modality and graph representations.
9. Gated fusion is combined with raw concatenation to form the final multimodal representation.
10. Layer normalization and dropout are applied before classification.
11. **MMILB and CPC objectives** are retained from the original MMIM framework.

## Training Objectives

The model retains the core learning objectives of the original MMIM framework:

- **Mutual Information Maximization (MMILB):** Learns shared information between text-audio and text-video representations.
- **Contrastive Predictive Coding (CPC):** Encourages the fused representation to preserve modality-specific information.
- **Sentiment Classification Loss:** Optimizes the final sentiment prediction.

## Results

| Metric | Original MMIM | Enhanced MMIM |
|---|---:|---:|
| MAE | **0.7635** | 0.7867 |
| Correlation | **0.7633** | 0.7439 |
| Mult Acc-7 | **0.4519** | 0.4359 |
| Mult Acc-5 | **0.5248** | 0.5015 |
| F1 Score | 0.8255 | **0.8332** |
| Accuracy | 0.8262 | **0.8323** |

The enhanced model improves **F1 Score and Accuracy** compared with the original MMIM, while showing a trade-off in MAE and correlation.

## Model Pipeline

```text
Input
  │
  ├── Text ──→ BERT ──→ Token Attention Pooling
  │
  ├── Audio ──→ RNN Encoder
  │
  └── Video ──→ RNN Encoder
                    │
                    ↓
              Common Projection
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
            Adaptive Gating
                    │
                    ↓
             Hybrid Fusion
                    │
                    ↓
          LayerNorm + Dropout
                    │
                    ↓
              Classification
```

## Project Structure

```text
Enhanced-MMIM/
│
├── modules/
│   ├── encoders.py
│   └── ...
│
├── data/
│   ├── CMU-MOSI/
│   └── CMU-MOSEI/
│
├── main.py
├── config.py
├── requirements.txt
└── README.md
```

## Key Contribution

This project extends the original MMIM framework by combining **mutual information learning with GCN-based cross-modal interaction, adaptive gated fusion, modality-level dropout, and Layer Normalization**, providing a more robust and adaptive framework for multimodal sentiment analysis.

## Reference

This implementation is based on the original **MMIM (Multimodal Mutual Information Maximization)** framework and extends its multimodal fusion architecture with additional graph-based, adaptive, and robustness mechanisms.
