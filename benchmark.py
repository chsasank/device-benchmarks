import torch
import time
import argparse
import numpy as np
try:
    import intel_extension_for_pytorch as ipex
    from torch import xpu
except ImportError:
    pass
from torch import mps, cuda



parser = argparse.ArgumentParser(description='Measure FLOPs and BW.')
parser.add_argument('--device', type=str, default='cpu',
                    help='One of cpu | cuda | mps | xpu')
parser.add_argument('--num_trails', type=int, default=10,
                    help='Number of trails to get average.')
parser.add_argument('--dtype', type=str, default='float',
                    help='One of float32|float64|float16|bfloat16|int8|int16|int32|bool')
args = parser.parse_args()

dtype = getattr(torch, args.dtype)
device = torch.device(args.device)
num_trails = args.num_trails


def flops_benchmark(device):
    test_range = 2 ** np.arange(8, 13, 0.25)

    print(f'benchmarking {device} using {dtype}')
    print('size, elapsed_time, tops')
    for n in test_range:
        total = 0
        for _ in range(num_trails):
            n = int(n)
            a = 10 * torch.rand(n, n, device=device)
            a = a.to(dtype)

            synchronize(device)
            now = time.time()
            b = torch.matmul(a, a)
            synchronize(device)

            total += time.time() - now

        total = total / num_trails

        tflops = 2 * n**3 / total / 1e12

        print(n, total, tflops, sep=", ")


def synchronize(device):
    if device.type == "cuda":
        cuda.synchronize()
    elif device.type == "mps":
        mps.synchronize()
    elif device.type == "xpu":
        xpu.synchronize()
    elif device.type == "cpu":
        pass


def memory_bandwidth_benchmark(device):
    test_range = 2 ** (np.arange(20, 27, 0.5))

    print('size (GB), elapsed_time, bandwidth (GB/s)')
    for size in test_range:
        elapsed_time = 0
        for _ in range(num_trails):
            size = int(size)

            # Create random tensors
            a = torch.rand(size, device=device)
            b = torch.rand(size, device=device)

            # Warm-up to ensure CUDA kernel is initialized if using GPU
            synchronize(device)
            a.copy_(b)
            synchronize(device)

            # Record the start time
            start_time = time.time()

            # Perform the copy operation
            a.copy_(b)

            # Synchronize if using CUDA to make sure operation is finished
            synchronize(device)

            # Record the end time
            end_time = time.time()

            # Compute elapsed time
            elapsed_time += end_time - start_time

        elapsed_time = elapsed_time / num_trails
        # Calculate Bandwidth in GB/s
        bytes_copied = a.nelement() * a.element_size()  # bytes
        bandwidth = 2 * bytes_copied / elapsed_time / 1e9  # GB/s

        print(bytes_copied / 1e9, elapsed_time, bandwidth, sep=', ')

    return bandwidth


if __name__ == "__main__":
    flops_benchmark(device)
    memory_bandwidth_benchmark(device)
