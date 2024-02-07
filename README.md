# "Attention is All You Need" Paper Implementation
  This provides an overview and guidance for implementing the "Attention is All You Need" paper by Vaswani et al. (2017). The paper introduces the Transformer architecture, a model based solely on self-attention mechanisms, achieving state-of-the-art results in various natural language processing tasks.

## Introduction
  The Transformer model introduced in the paper "Attention is All You Need" is a revolutionary neural network architecture that relies on the self-attention mechanism. This implementation aims to provide a clear and concise codebase for understanding and applying the Transformer model.

## Dependencies
  Install the required dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

## Model Architecture
  The model architecture is based on the Transformer described in the paper. The key components include:
  - Encoder: Composed of multiple layers, each containing a multi-head self-attention mechanism and a feedforward neural network.
  - Decoder: Similar to the encoder but also includes an additional multi-head attention layer to attend to the encoder's output.

  For a detailed explanation of the architecture, refer to Section 3 of the original paper.

## Training
  To train the Transformer model, run the following command:
  ```bash
  python trainer.py
  ```
  ![](https://i.imgur.com/xK0lClH.png)

## Inference
  This is a sample for a translator from English to Italian with this Transformer model:
  
  ![](https://i.imgur.com/jBqLpeA.png)
  
## References
 - [Opus Book](https://huggingface.co/datasets/opus_books)
 - [Pytorch-Transformers](https://github.com/hkproj/pytorch-transformer)
 - [Attention is all you need - Paper](https://arxiv.org/abs/1706.03762)
