# Device Benchmarks

Benchmarks of different devices I have come across. This repo is migrated from this gist here: https://gist.github.com/chsasank/407df67ac0c848d6259f0340887648a9#file-benchmark-py

I will maintain interesting benchmarks of different devices of I have come across.

## Matrix Multiplication FLOPS and BW

I have written a quick script in PyTorch to benchmark GPUs and CPUs. I use fp32 matrix multiplication to measure FLOPs (floating point operations per second). I copy a large tensor to measure bandwidth. These two are the most important metrics for LLM inference. Read [this blog](https://chsasank.com/llm-system-design.html) for more details on this.


Here's an example run:

```

```
