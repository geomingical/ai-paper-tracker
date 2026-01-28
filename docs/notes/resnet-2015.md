# Deep Residual Learning - Reading Notes

## Summary

ResNet introduced skip connections (residual connections) that allow training of extremely deep networks (100+ layers) by addressing the vanishing gradient problem.

## Key Innovation

**Residual Learning**: Instead of learning `H(x)`, learn `F(x) = H(x) - x`, then output `F(x) + x`

This simple change has profound implications:
- Easier to optimize (identity mapping is trivial to learn)
- Enables training of 152-layer networks
- Won ImageNet 2015 with 3.57% top-5 error

## Architecture Notes

- Bottleneck design: 1x1 → 3x3 → 1x1 convolutions
- Batch normalization after each convolution
- No dropout needed due to batch norm

## Impact

Skip connections are now ubiquitous - used in Transformers, U-Nets, and virtually all modern architectures.
