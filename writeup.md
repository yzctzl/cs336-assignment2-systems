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
