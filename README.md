# Device Benchmarks

Benchmarks of different devices I have come across. This repo is migrated from this gist here: https://gist.github.com/chsasank/407df67ac0c848d6259f0340887648a9#file-benchmark-py

I will maintain interesting benchmarks of different devices of I have come across.

## Matrix Multiplication FLOPS and BW

I have written a quick script in PyTorch to benchmark GPUs and CPUs. I use fp32 matrix multiplication to measure FLOPs (floating point operations per second). I copy a large tensor to measure bandwidth. These two are the most important metrics for LLM inference. Read [this blog](https://chsasank.com/llm-system-design.html) for more details on this.


Here's an example run:

```
(intel) sasank@ubuntu-22-04:~/code/device-benchmarks$ python benchmark.py --device xpu
benchmarking xpu
size, elapsed_time, flops
256, 0.011420178413391113, 0.00293817055963457
304, 0.0003251314163208008, 0.1728191284491545
362, 0.00033059120178222654, 0.28698844823613445
430, 0.0003793954849243164, 0.4191246504468045
512, 0.00037815570831298826, 0.7098543010167228
608, 0.008894515037536622, 0.05053804756110619
724, 0.0004009723663330078, 1.8929156014947033
861, 0.0005517244338989258, 2.3137542649304907
1024, 0.0006966352462768555, 3.0826514441770736
1217, 0.001168060302734375, 3.0862881116333902
1448, 0.001726818084716797, 3.516325684645487
1722, 0.0028204917907714844, 3.620800503449297
2048, 0.016068482398986818, 1.0691656347760172
2435, 0.008600807189941407, 3.35728090542124
2896, 0.013591170310974121, 3.5741173983212615
3444, 0.024279212951660155, 3.3649980718346795
4096, 0.03385140895843506, 4.060065967734354
4870, 0.06302995681762695, 3.6649653222576513
5792, 0.10398786067962647, 3.737085306267269
6888, 0.17345609664916992, 3.7680776333042645
size (GB), elapsed_time, bandwidth
0.004194304, 0.0003708839416503906, 22.61787868914374
0.00593164, 0.0004174232482910156, 28.42026659648161
0.008388608, 0.000445866584777832, 37.62833226975242
0.01186328, 0.0003901243209838867, 60.81794628994683
0.016777216, 0.00044062137603759763, 76.15252873509442
0.023726564, 0.0005816459655761719, 81.58421240486638
0.033554432, 0.0007857322692871094, 85.40932659019785
0.047453132, 0.0010800123214721679, 87.87516782274575
0.067108864, 0.0014967203140258789, 89.67455492000445
0.094906264, 0.002076077461242676, 91.42844211910285
0.134217728, 0.0029109954833984376, 92.21431552570304
0.189812528, 0.004096579551696777, 92.66878653504035
0.268435456, 0.005767607688903808, 93.0838123808032
0.37962506, 0.008129024505615234, 93.39990542229731

```

Some useful commands:

```
# for apple gpu
python benchmark.py --device mps --dtype float32

# for intel gpus with int8
python benchmark.py --device xpu --dtype int8

# for nvidia gpus with bfloat16
python benchmark.py --device cuda --dtype bfloat16
```


Here's a summary of the data I have collected for different devices

| Device | Device Type | TFLOPs (FP32) | TFLOPs (FP16)| TFLOPs (BF16) | TOPS (INT8) | Memory Bandwidth (GB/s) |
|---|---|---|---|---|---|---|
| Apple M1 CPU | CPU | 0.8 |  |  |  |  | 46 |
| Apple M1 GPU | GPU | 1.4 |  |  |  |  | 56 |
| Apple M1 Pro CPU 10-core | CPU | 0.3 |  |  | 0.008  | 96 |
| Apple M1 Pro GPU 16-core | GPU | 3.7 | 4.3 |  |  | 176 |
| Apple M2 CPU | CPU | 1 |    ||  | 60 |
| Apple M2 GPU | GPU | 2 |  | NA | NA | 90 |
| Apple M2 Ultra CPU | CPU | 4 |  |  |  | 311 |
| Apple M2 Ultra GPU (76 Core) | GPU | 20 |  |  |  | 636 |
| Apple M3 Max GPU (40 Core) | GPU | 11.4 |  |  |  | 318 |
| SteamDeck CPU | CPU | 0.17 | 0.002 | 0.002 | 0.05 | 20 |
| SteamDeck GPU | GPU | 1.22 | 2.2 | 0.5 | NA | 69 |
| Samsung Exynos 2100 | CPU | 0.1 |  |  |  | 16 |
| AMD Ryzen 5 3600 | CPU | 0.36 |  |  |  | 14 |
| AMD Ryzen 5 4600HS | CPU | 0.4 |  |  |  | 22 |
| AMD Ryzen 9 5900X | CPU | 1.3 |  |  |  | 29 |
| AMD Ryzen 9 7950X | CPU | 1.1 |  |  |  | 28 |
| AMD Ryzen Threadripper 3960X 24-Cores | CPU | 1.4 |  |  |  | 44 |
| AMD Ryzen Threadripper PRO 5975WX 32-Cores | CPU | 1.5 |  |  |  | 28 |
| AMD Epyc 7763 Engineering Sample | CPU | 3.2 |  |  |  | 115 |
| AMD Epyc 7262 | CPU | 0.5 |  |  |  | 80 |
| Intel i5-12400 | CPU | 0.7 |  | 0.003 | 0.05 | 26 |
| Intel i7-8559U | CPU | 0.2 |  |  |  | 10 |
| Intel i7-8750H | CPU | 0.5 |  |  |  | 15 |
| Intel i7-1360P | CPU | 0.4 |  | 0.003 | 0.06 | 24 |
| Intel i9-13900K (WSL2) | CPU | 1.2 |  |  |  | 49 |
| Intel Xeon Silver 4116 | CPU | 0.5 |  |  |  | 20 |
| Intel Xeon Gold 6230 | CPU | 1.9 | NA | 0.61 | 0.014 | 17.5 |
| Intel Xeon Gold 6330 | CPU | 5.7 | NA | 0.75 | 0.02 | 81 |
| Intel Xeon Platinum 8358 | CPU | 3.5 |  | 0.96 | 0.029  | 96 |
| Intel Xeon Platinum 8358 | CPU | 5.6 | NA | 14 | 0.04 | 137 |
| AMD 7900 XTX | GPU | 26 | 101 | 104 | NA | 792 |
| Intel Arc 770 16GB | GPU | 15 | 86 | 90 | 174 | 452 |
| Intel Arc 370m | GPU | 4 |  | 15 | 35 | 93 |
| Intel Data Center GPU Max 1100 | GPU | 21 | 140 | 140 | 221 | 781 |
| Nvidia T4 | GPU | 4 |  | 2.25 | NA | 240 |
| Nvidia V100 32GB | GPU | 13 | 84 | 9.4 | NA  | 766 |
| Nvidia A10 24GB | GPU | 14 | 54 | 56 | NA | 469 |
| Nvidia A100 80GB | GPU | 19 | 189 | 237 | NA | 1490 |
| Nvidia H100-PCIe 80GB | GPU | 38 | 435 | 449 | NA  | 1630 |
| Nvidia 1050 Ti Mobile | GPU | 1.8 |1.5  | 1 | NA | 97 |
| Nvidia 1060 Ti Mobile | GPU | 3.8 | 17.6  | 2.18 | NA | 222 |
| Nvidia 1650 Ti Mobile | GPU | 3 |  | 1.8 | NA | 172 |
| Nvidia 2070S | GPU | 8 | 37 | 5 | NA | 831 |
| Nvidia 3090 | GPU | 27 |  |  |  | 831 |
| Nvidia 4060ti | GPU | 12 | 42 | 46 | NA | 234 |
| Nvidia 4070 Super | GPU | 23 |  |  |  | 411 |
| Nvidia 4090 | GPU | 58 | 150 | 168 | NA | 912 |
| Nvidia 4090 (WSL2) | GPU | 53 |  |  |  | 885 |


NA = not available on the device. Usually shows up as error like these:

```
RuntimeError: "addmm_cuda" not implemented for 'Char'
RuntimeError: MPS device does not support mm for non-float inputs
```
