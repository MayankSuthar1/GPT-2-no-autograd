# Input
```
python train_gpt_v3.py
testing prompt : "Hello There! How are you doing today?"
```

# Output
## Final output
```
--- Running Downstream Benchmark Eval ---
Downstream Task - Loss: 2.6295 | Perplexity: 13.87


--- sample ---
Hi there! How are you doing today?

That seens ble, buch teallin:
You lorses your thath the the land so me sure,
And mother my astare whe fall dand stily.

RORGLO:
So ast pich is outh astee plivere wor my seraper ande dive broty.

No 
```
## Logs
```
[DEBUG] Starting the main training script.
[DEBUG] Ensuring dataset 'shakespeare_char' is available and loading data.
[DEBUG] Creating output directory: out-gpt-ffn-bp-v3-shakespeare-char
[DEBUG] Initializing the GPT2 model architecture.
[DEBUG] Initializing GPT2 model: vocab_size=65, n_embd=64, n_head=2, n_layer=2
[DEBUG] Collecting parameters for the optimizer.
dataset: data\shakespeare_char
vocab size: 65
model parameters: 112,577
tokens per iteration: 512
[DEBUG] Entering the training loop for 2000 iterations.
step 0: train loss 4.2099 (PPL: 67.35, BPC: 6.07) | val loss 4.2111 (PPL: 67.43, BPC: 6.08)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 0: loss 4.1916, lr 2.00e-05, grad_norm 2.5979, time 2913.29ms
step 100: train loss 2.6615 (PPL: 14.32, BPC: 3.84) | val loss 2.6668 (PPL: 14.39, BPC: 3.85)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 100: loss 2.6099, lr 9.99e-04, grad_norm 1.1197, time 16188.54ms
step 200: train loss 2.5501 (PPL: 12.81, BPC: 3.68) | val loss 2.5552 (PPL: 12.87, BPC: 3.69)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 200: loss 2.5600, lr 9.87e-04, grad_norm 1.8152, time 16484.96ms
step 300: train loss 2.5043 (PPL: 12.23, BPC: 3.61) | val loss 2.4923 (PPL: 12.09, BPC: 3.60)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 300: loss 2.4310, lr 9.64e-04, grad_norm 1.7587, time 16207.79ms
step 400: train loss 2.4227 (PPL: 11.28, BPC: 3.50) | val loss 2.4510 (PPL: 11.60, BPC: 3.54)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 400: loss 2.4625, lr 9.30e-04, grad_norm 1.2873, time 16403.75ms
step 500: train loss 2.4017 (PPL: 11.04, BPC: 3.46) | val loss 2.4195 (PPL: 11.24, BPC: 3.49)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 500: loss 2.5235, lr 8.87e-04, grad_norm 1.1787, time 16573.98ms
step 600: train loss 2.3697 (PPL: 10.69, BPC: 3.42) | val loss 2.4111 (PPL: 11.15, BPC: 3.48)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 600: loss 2.3646, lr 8.35e-04, grad_norm 1.3695, time 16389.26ms
step 700: train loss 2.3501 (PPL: 10.49, BPC: 3.39) | val loss 2.3600 (PPL: 10.59, BPC: 3.40)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 700: loss 2.3466, lr 7.75e-04, grad_norm 2.7442, time 16341.39ms
step 800: train loss 2.3151 (PPL: 10.13, BPC: 3.34) | val loss 2.3098 (PPL: 10.07, BPC: 3.33)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 800: loss 2.4060, lr 7.10e-04, grad_norm 1.5578, time 16455.82ms
step 900: train loss 2.2731 (PPL: 9.71, BPC: 3.28) | val loss 2.2770 (PPL: 9.75, BPC: 3.28)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 900: loss 2.3215, lr 6.40e-04, grad_norm 2.0072, time 16411.43ms
step 1000: train loss 2.2071 (PPL: 9.09, BPC: 3.18) | val loss 2.2747 (PPL: 9.73, BPC: 3.28)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 1000: loss 2.2569, lr 5.68e-04, grad_norm 1.9719, time 16390.67ms
step 1100: train loss 2.2042 (PPL: 9.06, BPC: 3.18) | val loss 2.2210 (PPL: 9.22, BPC: 3.20)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 1100: loss 2.1957, lr 4.96e-04, grad_norm 2.7358, time 16094.38ms
step 1200: train loss 2.1780 (PPL: 8.83, BPC: 3.14) | val loss 2.2053 (PPL: 9.07, BPC: 3.18)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 1200: loss 2.2076, lr 4.25e-04, grad_norm 2.7578, time 16273.29ms
step 1300: train loss 2.1414 (PPL: 8.51, BPC: 3.09) | val loss 2.1959 (PPL: 8.99, BPC: 3.17)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 1300: loss 2.0530, lr 3.57e-04, grad_norm 2.9836, time 16232.01ms
step 1400: train loss 2.1196 (PPL: 8.33, BPC: 3.06) | val loss 2.1745 (PPL: 8.80, BPC: 3.14)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 1400: loss 2.2623, lr 2.94e-04, grad_norm 2.7727, time 16308.72ms
step 1500: train loss 2.1073 (PPL: 8.23, BPC: 3.04) | val loss 2.1443 (PPL: 8.54, BPC: 3.09)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 1500: loss 2.1513, lr 2.38e-04, grad_norm 3.2522, time 16511.93ms
step 1600: train loss 2.0815 (PPL: 8.02, BPC: 3.00) | val loss 2.1446 (PPL: 8.54, BPC: 3.09)
iter 1600: loss 2.0157, lr 1.90e-04, grad_norm 3.0770, time 16176.55ms
step 1700: train loss 2.0735 (PPL: 7.95, BPC: 2.99) | val loss 2.1484 (PPL: 8.57, BPC: 3.10)
iter 1700: loss 2.1534, lr 1.52e-04, grad_norm 3.2900, time 16384.82ms
step 1800: train loss 2.0540 (PPL: 7.80, BPC: 2.96) | val loss 2.1186 (PPL: 8.32, BPC: 3.06)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 1800: loss 2.0113, lr 1.23e-04, grad_norm 3.0396, time 16568.15ms
step 1900: train loss 2.0474 (PPL: 7.75, BPC: 2.95) | val loss 2.1139 (PPL: 8.28, BPC: 3.05)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 1900: loss 2.0212, lr 1.06e-04, grad_norm 3.0232, time 16233.52ms
step 2000: train loss 2.0653 (PPL: 7.89, BPC: 2.98) | val loss 2.1035 (PPL: 8.19, BPC: 3.03)
saved best checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_best_v3.npz
iter 2000: loss 1.9536, lr 1.00e-04, grad_norm 4.3796, time 16125.82ms
saved last checkpoint to out-gpt-ffn-bp-v3-shakespeare-char\ckpt_last_v3.npz


```

