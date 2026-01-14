# cs336 Assignment2 Systems

## benchmarking_script
| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |   Total(ms) |   CV(%) |
|:--------|----------:|--------------:|---------------:|-----------:|------------:|--------:|
| small   |       128 |        21.008 |         32.322 |      8.268 |      61.598 |   5.861 |
| medium  |       128 |        39.344 |         59.057 |     16.543 |     114.945 |   3.283 |
| large   |       128 |        62.067 |         92.596 |     36.561 |     191.224 |   1.420 |
| xl      |       128 |        99.746 |        120.695 |     90.808 |     311.249 |   0.723 |
| 2.7B    |       128 |       nan     |        nan     |    nan     |     nan     | nan     |

| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |   Total(ms) |   CV(%) |
|:--------|----------:|--------------:|---------------:|-----------:|------------:|--------:|
| small   |       256 |        17.009 |         24.925 |      5.977 |      47.911 |   8.394 |
| medium  |       256 |        35.707 |         49.744 |     13.585 |      99.036 |   4.539 |
| large   |       256 |        68.879 |         92.775 |     35.064 |     196.718 |   1.347 |
| xl      |       256 |       120.358 |        163.762 |     89.978 |     374.099 |   4.842 |
| 2.7B    |       256 |       nan     |        nan     |    nan     |     nan     | nan     |

| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |   Total(ms) |   CV(%) |
|:--------|----------:|--------------:|---------------:|-----------:|------------:|--------:|
| small   |       512 |        22.067 |         40.793 |      7.183 |      70.043 |   6.094 |
| medium  |       512 |        59.422 |        107.057 |     13.587 |     180.067 |   0.783 |
| large   |       512 |       114.608 |        199.370 |     35.730 |     349.708 |   0.257 |
| xl      |       512 |       nan     |        nan     |    nan     |     nan     | nan     |
| 2.7B    |       512 |       nan     |        nan     |    nan     |     nan     | nan     |

| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |   Total(ms) |   CV(%) |
|:--------|----------:|--------------:|---------------:|-----------:|------------:|--------:|
| small   |      1024 |        47.671 |        106.880 |      5.111 |     159.662 |   0.952 |
| medium  |      1024 |       138.719 |        290.854 |      9.994 |     439.567 |   0.095 |
| large   |      1024 |       nan     |        nan     |    nan     |     nan     | nan     |
| xl      |      1024 |       nan     |        nan     |    nan     |     nan     | nan     |
| 2.7B    |      1024 |       nan     |        nan     |    nan     |     nan     | nan     |


change batch size to 2

| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |   Total(ms) |   CV(%) |
|:--------|----------:|--------------:|---------------:|-----------:|------------:|--------:|
| small   |       512 |        16.647 |         26.909 |      7.160 |      50.716 |   9.692 |
| medium  |       512 |        44.428 |         65.104 |     10.191 |     119.724 |   2.467 |
| large   |       512 |        80.356 |        119.124 |     35.034 |     234.514 |   0.629 |
| xl      |       512 |       132.934 |        202.117 |     89.213 |     424.263 |   0.718 |
| 2.7B    |       512 |       nan     |        nan     |    nan     |     nan     | nan     |

| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |   Total(ms) |   CV(%) |
|:--------|----------:|--------------:|---------------:|-----------:|------------:|--------:|
| small   |      1024 |        27.442 |         57.824 |      5.216 |      90.482 |   2.522 |
| medium  |      1024 |        81.680 |        153.266 |     10.645 |     245.591 |   0.325 |
| large   |      1024 |       150.984 |        289.782 |     36.049 |     476.815 |   0.381 |
| xl      |      1024 |       nan     |        nan     |    nan     |     nan     | nan     |
| 2.7B    |      1024 |       nan     |        nan     |    nan     |     nan     | nan     |

b. no high variability across measurements, but larger model/batch_size has more stable Coefficient of Variation


| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |   Total(ms) |   CV(%) |
|:--------|----------:|--------------:|---------------:|-----------:|------------:|--------:|
| small   |       128 |        16.572 |         25.294 |      5.861 |      47.726 |   7.613 |
| medium  |       128 |        31.145 |         48.424 |     12.498 |      92.066 |   5.248 |
| large   |       128 |        56.169 |         72.867 |     35.262 |     164.298 |   2.533 |
| xl      |       128 |        90.623 |        101.603 |     91.539 |     283.765 |   1.319 |
| 2.7B    |       128 |       nan     |        nan     |    nan     |     nan     | nan     |

| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |   Total(ms) |   CV(%) |
|:--------|----------:|--------------:|---------------:|-----------:|------------:|--------:|
| small   |       256 |        22.015 |         31.798 |      7.211 |      61.024 |   8.312 |
| medium  |       256 |        40.015 |         60.672 |     13.940 |     114.627 |   4.828 |
| large   |       256 |        68.543 |         93.877 |     36.756 |     199.176 |   1.936 |
| xl      |       256 |       121.884 |        164.609 |     90.563 |     377.055 |   6.395 |
| 2.7B    |       256 |       nan     |        nan     |    nan     |     nan     | nan     |

| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |   Total(ms) |   CV(%) |
|:--------|----------:|--------------:|---------------:|-----------:|------------:|--------:|
| small   |       512 |        20.189 |         40.372 |      5.083 |      65.644 |   3.485 |
| medium  |       512 |        62.551 |        106.171 |     10.411 |     179.133 |   0.929 |
| large   |       512 |       114.821 |        198.050 |     35.431 |     348.302 |   0.294 |
| xl      |       512 |       nan     |        nan     |    nan     |     nan     | nan     |
| 2.7B    |       512 |       nan     |        nan     |    nan     |     nan     | nan     |

| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |   Total(ms) |   CV(%) |
|:--------|----------:|--------------:|---------------:|-----------:|------------:|--------:|
| small   |      1024 |        45.748 |        107.716 |      7.084 |     160.549 |   0.740 |
| medium  |      1024 |       130.305 |        292.417 |     18.580 |     441.302 |   0.201 |
| large   |      1024 |       nan     |        nan     |    nan     |     nan     | nan     |
| xl      |      1024 |       nan     |        nan     |    nan     |     nan     | nan     |
| 2.7B    |      1024 |       nan     |        nan     |    nan     |     nan     | nan     |

change batch size to 2

| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |   Total(ms) |   CV(%) |
|:--------|----------:|--------------:|---------------:|-----------:|------------:|--------:|
| small   |       512 |        15.190 |         32.755 |      5.126 |      53.071 |   4.359 |
| medium  |       512 |        44.592 |         65.183 |     10.382 |     120.157 |   2.733 |
| large   |       512 |        80.480 |        119.242 |     35.353 |     235.076 |   1.021 |
| xl      |       512 |       133.015 |        203.522 |     89.746 |     426.283 |   0.574 |
| 2.7B    |       512 |       nan     |        nan     |    nan     |     nan     | nan     |

| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |   Total(ms) |   CV(%) |
|:--------|----------:|--------------:|---------------:|-----------:|------------:|--------:|
| small   |      1024 |        27.042 |         58.940 |      6.031 |      92.013 |   3.070 |
| medium  |      1024 |        82.065 |        154.407 |     10.561 |     247.033 |   0.210 |
| large   |      1024 |       151.293 |        290.324 |     36.172 |     477.789 |   0.482 |
| xl      |      1024 |       nan     |        nan     |    nan     |     nan     | nan     |
| 2.7B    |      1024 |       nan     |        nan     |    nan     |     nan     | nan     |


c. nearly same as with warm up, slightly high CV

## nsys_profile

| Model   |   Context |   Forward(ms) |   Backward(ms) |   Step(ms) |
|:--------|----------:|--------------:|---------------:|-----------:|
| small   |       512 |        30.130 |         48.853 |     17.390 |
| medium  |       512 |        50.162 |         84.061 |     29.585 |
| large   |       512 |       108.094 |        177.765 |     65.187 |
| xl      |       512 |       nan     |        nan     |    nan     |
| 2.7B    |       512 |       nan     |        nan     |    nan     |

a. fairly match with the Python standard library measured results

b. `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` at 109 instances, not same kernel when both forward and backward passes, it's `elementwise_kernel`

c. `at::native::vectorized_elementwise_kernel`, `at::native::elementwise_kernel` and more other element wise tensor operation

d. the fraction of time spent on matrix multiplication change to 16.5% and `vectorized_elementwise_kernel` operations change to 35%+. element wise operations include: SwiGLU, RMSNorm Scaling, Add, RoPE, Activation Grad, Norm Grad, AdamW update etc.

e. softmax takes 2x times of mutmal in dot self attention. softmax is memory bound, need read/write entire matrix from HBM. FLOPS of mutmal: `2*L^2*d` softmax: `3*L^2*H`, $FLOPS(mutmal)/FLOPS(softmax) = 2d/3H = 2*d_{head}/3$


## mixed_precision_accumulation

save as fp32 and compute with fp16 with acceptable precesion loss, when save/compute are fp16 the calculation error will be 20x larger than save as fp32.


## benchmarking_mixed_precision

a. 
the model parameters within the autocast context: FP32
the output of the first feed-forward layer (ToyModel.fc1): FP16
the output of layer norm (ToyModel.ln): FP32
the model’s predicted logits: FP16
the loss: FP32
and the model’s gradients: FP32

b. same as fp16, bf16 still need to treat layer normalization differently as fp32, the sensitive part of LN is standardization and epsion.

c. train with bf16 save ~35% time, finally get fairly same result


## memory_profiling
**limited by 40GiB GPU memory, use large size model**

a. 
![large_forward_pass](images/large_forward_pass.png)
![large_one_step](images/large_one_step.png)

stage1: forward pass calculate and save activations to memory
stage2: backward pass calculate gradients and free activations
stage3: optimizer step don't need new memory, i implemented in-place

b. 
| Model   |   Context |  Memory  |
|:--------|----------:|---------:|
| large   |       512 |   27.9G  |
| large   |       256 |   18.9G  |
| large   |       128 |   15.2G  |

for one training step same as forward pass's peak memory

c.
| Model   |   Context |  Memory  |
|:--------|----------:|---------:|
| large   |       512 |   32.2G  |
| large   |       256 |   19.8G  |
| large   |       128 |   15.8G  |

when context length is long, mixed-precision significantly affect memory usage, bcuz in this case the activation is huge and mixed-precision save this part as bf16.

d. 
| Model   |   Context | residual |
|:--------|----------:|---------:|
| large   |       512 |   10MB   |
| large   |       256 |    5MB   |
| large   |       128 |   2.5MB  |

e.
largest allocate memory is 80MB from scaled_dot_product_attention.


## pytorch_attention
|d_model    | seq_len    | Forward (ms)    | Backward (ms)   | Memory (MB) |
|:----------|-----------:|----------------:|----------------:|------------:|
|16         | 256        | 0.2115          | 0.7100          | 23.14       |
|16         | 1024       | 0.4067          | 1.3922          | 115.81      |
|16         | 4096       | 4.8074          | 15.0105         | 1566.50     |
|16         | 8192       | 18.6628         | 58.4936         | 6188.75     |
|16         | 16384      | OOM             | OOM             | OOM         |
|32         | 256        | 0.2299          | 0.6306          | 24.02       |
|32         | 1024       | 0.3913          | 1.3267          | 119.31      |
|32         | 4096       | 4.9703          | 15.1978         | 1580.50     |
|32         | 8192       | 18.7202         | 58.5669         | 6216.75     |
|32         | 16384      | OOM             | OOM             | OOM         |
|64         | 256        | 0.2247          | 0.5972          | 25.77       |
|64         | 1024       | 0.4022          | 1.3476          | 126.31      |
|64         | 4096       | 5.0355          | 15.3106         | 1608.50     |
|64         | 8192       | 19.0098         | 58.9260         | 6272.75     |
|64         | 16384      | OOM             | OOM             | OOM         |
|128        | 256        | 0.2375          | 0.5643          | 29.27       |
|128        | 1024       | 0.4301          | 1.4106          | 140.31      |
|128        | 4096       | 5.1718          | 15.6606         | 1664.50     |
|128        | 8192       | 19.5993         | 60.1373         | 6384.75     |
|128        | 16384      | OOM             | OOM             | OOM         |

a. at all d_model configurations with 16384 sequence length get OOM

b. 
```
forward and backward memory of scaled_dot_product_attention d_model:128 seq_len 8192
= Parameters + Gradients + Activations
= 2 * (3 * batch_size * context_length * d_model) + 2 * (batch_size * context_length * context_length)
= 4.38 GiB
```

c. in backward, bynamic release the activatons and compute the gradients, the activation is $O(L^2)$ but gradients is $O(L)$

d. 1. FlashAttention 2. Gradients checkpoints 3. ZeRO 4. operator fusion


## torch_compile
a.
|d_model    | seq_len    | Forward (ms)    | Backward (ms)   | Memory (MB) |
|:----------|-----------:|----------------:|----------------:|------------:|
|16         | 256        | 0.1788          | 0.5356          | 21.14       |
|16         | 1024       | 0.2219          | 0.6478          | 83.81       |
|16         | 4096       | 1.7815          | 3.3749          | 1054.50     |
|16         | 8192       | 8.1544          | 18.2337         | 4140.75     |
|16         | 16384      | 31.8973         | 72.5969         | 16457.25    |
|32         | 256        | 0.2278          | 0.6499          | 22.02       |
|32         | 1024       | 0.2523          | 0.6410          | 87.31       |
|32         | 4096       | 2.5957          | 4.8566          | 1068.50     |
|32         | 8192       | 9.8550          | 19.0695         | 4168.75     |
|32         | 16384      | 32.3524         | 75.7464         | 16513.25    |
|64         | 256        | 0.2328          | 0.6402          | 23.77       |
|64         | 1024       | 0.3388          | 0.5769          | 94.31       |
|64         | 4096       | 2.0023          | 3.6305          | 1096.50     |
|64         | 8192       | 7.0231          | 14.9141         | 4224.75     |
|64         | 16384      | OOM             | OOM             | OOM         |
|128        | 256        | 0.2281          | 0.6325          | 27.27       |
|128        | 1024       | 0.3554          | 0.6079          | 108.31      |
|128        | 4096       | 2.1769          | 4.0207          | 1152.50     |
|128        | 8192       | 7.7799          | 16.3734         | 4336.75     |
|128        | 16384      | OOM             | OOM             | OOM         |

b.
| Model      | Mode       | Fwd (ms)   | Bwd (ms)   | Opt (ms)   | Step (ms)  | Mem (MB)   |
|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|
| small      | Vanilla    | 17.34      | 27.41      | 20.80      | 65.56      | 2259.56    |
| small      | Compiled   | 7.69       | 12.74      | 9.83       | 30.25      | 2092.75    |
| medium     | Vanilla    | 35.28      | 51.41      | 30.30      | 116.99     | 6831.00    |
| medium     | Compiled   | 15.72      | 24.38      | 30.93      | 71.03      | 6536.96    |
| large      | Vanilla    | 53.58      | 78.58      | 71.85      | 204.01     | 15218.19   |
| large      | Compiled   | 18.22      | 38.47      | 69.69      | 126.38     | 15063.04   |
| xl         | Vanilla    | 64.80      | 113.84     | 144.63     | 323.27     | 30869.83   |
| xl         | Compiled   | 28.38      | 70.86      | 143.57     | 242.81     | 30900.49   |
| 2.7B       | Vanilla    | OOM        | OOM        | OOM        | OOM        | OOM        |
| 2.7B       | Compiled   | OOM        | OOM        | OOM        | OOM        | OOM        |



