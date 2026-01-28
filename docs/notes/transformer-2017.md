# Attention Is All You Need - Reading Notes

## Key Takeaways

The Transformer architecture completely revolutionized NLP by replacing recurrent networks with self-attention mechanisms.

## Core Concepts

### Self-Attention Mechanism
- Query, Key, Value paradigm
- Scaled dot-product attention: `Attention(Q,K,V) = softmax(QK^T / √d_k)V`
- Multi-head attention allows the model to attend to different representation subspaces

### Architecture Highlights
1. **Encoder-Decoder structure** but without recurrence
2. **Positional encoding** to inject sequence order information
3. **Layer normalization** and residual connections throughout

## Personal Insights

This paper is foundational - understanding it deeply unlocks understanding of GPT, BERT, and virtually all modern LLMs.

## Questions to Explore
- [ ] Why does scaling by √d_k help with gradient stability?
- [ ] How does multi-head attention compare to single-head with larger dimensions?

## Related Papers
- BERT (bidirectional pre-training)
- GPT series (decoder-only architecture)
