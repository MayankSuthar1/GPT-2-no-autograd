"""
Train gpt_ffn_bp_v3.GPT2 on Tiny Shakespeare without PyTorch autograd.

Default usage from the repo root:
  python no_autograd/train_gpt_ffn_bp_v3.py

Edit the config values near the top of this file to change the dataset,
model size, optimizer, evaluation, checkpoint, or sampling settings.
"""

import json
import math 
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt # [ADDED] For generating artifact plots

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from gpt_ffn_bp_v3 import CrossEntropyLoss, GPT2


# -----------------
# Single-file config
# -----------------

# I/O and reproducibility
out_dir = "out-gpt-ffn-bp-v3-shakespeare-char"
seed = 1337
eval_interval = 100
log_interval = 100
eval_iters = 20
eval_only = False
always_save_checkpoint = False

# data
dataset = "shakespeare_char"
prepare_data = True
gradient_accumulation_steps = 1
batch_size = 8
block_size = 64

# model
n_layer = 2
n_head = 2
n_embd = 64
vocab_size = 50304

# optimizer
learning_rate = 1e-3
max_iters = 2000
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 50
lr_decay_iters = 2000
min_lr = 1e-4

# sampling
sample_tokens = 200
sample_prompt = "Hi there! How are you doing today?"
temperature = 0.8

# logging
wandb_log = False
wandb_project = "no-autograd-gpt"
wandb_run_name = "gpt-ffn-bp-v3"

class ParameterRef:
    def __init__(self, key, owner, value_name, grad_name, weight_decay):
        self.key = key
        self.owner = owner
        self.value_name = value_name
        self.grad_name = grad_name
        self.weight_decay = weight_decay

    @property
    def data(self):
        return getattr(self.owner, self.value_name)

    @property
    def grad(self):
        return getattr(self.owner, self.grad_name, None)

    def zero_grad(self):
        setattr(self.owner, self.grad_name, np.zeros_like(self.data))


class AdamW:
    def __init__(self, parameters, lr, betas, weight_decay, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]

    def set_lr(self, lr):
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            grad = p.grad
            if grad is None:
                continue

            if p.weight_decay and self.weight_decay != 0.0:
                p.data[...] *= 1.0 - self.lr * self.weight_decay

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            p.data[...] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def collect_parameters(model):
    params = []

    def add_embedding(key, layer):
        params.append(ParameterRef(f"{key}_weight", layer, "weight", "grad_weight", True))

    def add_linear(key, layer):
        params.append(ParameterRef(f"{key}_weight", layer, "weight", "grad_weight", True))
        params.append(ParameterRef(f"{key}_bias", layer, "bias", "grad_bias", False))

    def add_layer_norm(key, layer):
        params.append(ParameterRef(f"{key}_gamma", layer, "gamma", "grad_gamma", False))
        params.append(ParameterRef(f"{key}_beta", layer, "beta", "grad_beta", False))

    add_embedding("token_emb", model.token_emb)
    add_embedding("pos_emb", model.pos_emb)
    for block_idx, block in enumerate(model.blocks):
        prefix = f"blocks_{block_idx}"
        add_layer_norm(f"{prefix}_ln1", block.ln1)
        for head_idx, head in enumerate(block.attn.heads):
            head_prefix = f"{prefix}_attn_heads_{head_idx}"
            add_linear(f"{head_prefix}_q_proj", head.q_proj)
            add_linear(f"{head_prefix}_k_proj", head.k_proj)
            add_linear(f"{head_prefix}_v_proj", head.v_proj)
        add_linear(f"{prefix}_attn_proj", block.attn.proj)
        add_layer_norm(f"{prefix}_ln2", block.ln2)
        add_linear(f"{prefix}_ffn_fc1", block.ffn.fc1)
        add_linear(f"{prefix}_ffn_fc2", block.ffn.fc2)
    add_layer_norm("ln_f", model.ln_f)
    add_linear("lm_head", model.lm_head)
    return params


def clip_grad_norm(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    total_sq = 0.0
    for grad in grads:
        grad64 = grad.astype(np.float64, copy=False)
        total_sq += float(np.sum(grad64 * grad64))
    total_norm = math.sqrt(total_sq)

    if max_norm > 0.0 and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for grad in grads:
            grad[...] *= scale
    return total_norm


def ensure_dataset(dataset, prepare_data):
    data_dir = ROOT_DIR / "data" / dataset
    required = [data_dir / "train.bin", data_dir / "val.bin"]
    if all(path.exists() for path in required):
        return data_dir

    prepare_script = data_dir / "prepare.py"
    if not prepare_data or not prepare_script.exists():
        missing = ", ".join(str(path.relative_to(ROOT_DIR)) for path in required if not path.exists())
        raise FileNotFoundError(f"missing dataset files: {missing}")

    print(f"preparing dataset with {prepare_script.relative_to(ROOT_DIR)}")
    subprocess.run([sys.executable, str(prepare_script)], cwd=ROOT_DIR, check=True)
    if not all(path.exists() for path in required):
        missing = ", ".join(str(path.relative_to(ROOT_DIR)) for path in required if not path.exists())
        raise FileNotFoundError(f"dataset preparation did not create: {missing}")
    return data_dir


def load_dataset(data_dir, default_vocab_size):
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    meta_path = data_dir / "meta.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    else:
        meta = {"vocab_size": default_vocab_size}
    return train_data, val_data, meta


def display_path(path):
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def get_batch(data, batch_size, block_size, rng):
    if len(data) <= block_size + 1:
        raise ValueError("dataset is too small for the requested block_size")
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])
    return x, y


def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, rng):
    out = {}
    loss_fn = CrossEntropyLoss()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, rng)
            logits = model.forward(x)
            losses.append(loss_fn.forward(logits, y))
        mean_loss = float(np.mean(losses))
        
        # [ADDED] Calculate Perplexity and BPC (Bits per character)
        out[split] = {
            "loss": mean_loss,
            "perplexity": math.exp(mean_loss),
            "bpc": mean_loss / math.log(2) 
        }
    return out


# [ADDED] Downstream evaluation stub
def evaluate_downstream(model, meta, block_size, batch_size, rng):
    """
    Evaluates the model on a simulated downstream benchmarking task.
    Since this is a char-level model, we simulate a benchmark by 
    evaluating perplexity on a completely separate pseudo-corpus.
    """
    print("\n--- Running Downstream Benchmark Eval ---")
    # Simulate a downstream task dataset (e.g., a tiny slice of Wikipedia)
    dummy_downstream_text = "The quick brown fox jumps over the lazy dog. " * 50 
    
    stoi = meta.get("stoi", {})
    if not stoi:
        print("Skipping downstream eval: No string-to-int mapping found in meta.")
        return None
        
    # Tokenize the downstream text
    data = np.array([stoi.get(ch, 0) for ch in dummy_downstream_text], dtype=np.uint16)
    
    if len(data) <= block_size + 1:
        print("Downstream corpus too small for block size.")
        return None

    # Evaluate
    loss_fn = CrossEntropyLoss()
    losses = []
    # Run a few evaluation iterations
    for _ in range(10): 
        x, y = get_batch(data, batch_size, block_size, rng)
        logits = model.forward(x)
        losses.append(loss_fn.forward(logits, y))
        
    mean_loss = float(np.mean(losses))
    downstream_metrics = {
        "benchmark_loss": mean_loss,
        "benchmark_perplexity": math.exp(mean_loss)
    }
    
    print(f"Downstream Task - Loss: {downstream_metrics['benchmark_loss']:.4f} | Perplexity: {downstream_metrics['benchmark_perplexity']:.2f}\n")
    return downstream_metrics


# [ADDED] Plotting Artifacts
def plot_metrics(history, output_dir):
    """Generates and saves a training artifact plot (Loss and Perplexity)"""
    if not history["iter"]:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Plot
    ax1.plot(history["iter"], history["train_loss"], label="Train Loss", color="blue")
    ax1.plot(history["iter"], history["val_loss"], label="Val Loss", color="orange")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Perplexity Plot
    ax2.plot(history["iter"], history["train_ppl"], label="Train PPL", color="blue", linestyle="--")
    ax2.plot(history["iter"], history["val_ppl"], label="Val PPL", color="orange", linestyle="--")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Model Perplexity (Lower is better)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = output_dir / "training_curves.png"
    plt.savefig(plot_path)
    plt.close()


def get_lr(iter_num, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / max(1, warmup_iters)
    if iter_num > lr_decay_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def save_checkpoint(path, parameters, iter_num, best_val_loss):
    arrays = {p.key: p.data for p in parameters}
    arrays["iter_num"] = np.array(iter_num, dtype=np.int64)
    arrays["best_val_loss"] = np.array(best_val_loss, dtype=np.float64)
    np.savez(path, **arrays)


def sample_text(model, meta, prompt, block_size, max_new_tokens, temperature, rng):
    stoi = meta["stoi"]
    itos = meta["itos"]
    ids = [stoi[ch] for ch in prompt if ch in stoi]
    if not ids:
        ids = [0]

    for _ in range(max_new_tokens):
        x = np.array([ids[-block_size:]], dtype=np.int64)
        logits = model.forward(x)[0, -1, :] / max(temperature, 1e-6)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        ids.append(int(rng.choice(len(probs), p=probs)))
    return "".join(itos[int(i)] for i in ids)


def build_config_dict():
    return {
        "out_dir": out_dir,
        "seed": seed,
        "eval_interval": eval_interval,
        "log_interval": log_interval,
        "eval_iters": eval_iters,
        "eval_only": eval_only,
        "always_save_checkpoint": always_save_checkpoint,
        "dataset": dataset,
        "prepare_data": prepare_data,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "batch_size": batch_size,
        "block_size": block_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
        "grad_clip": grad_clip,
        "decay_lr": decay_lr,
        "warmup_iters": warmup_iters,
        "lr_decay_iters": lr_decay_iters,
        "min_lr": min_lr,
        "sample_tokens": sample_tokens,
        "sample_prompt": sample_prompt,
        "temperature": temperature,
        "wandb_log": wandb_log,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
    }


def main():
    if n_embd % n_head != 0:
        raise ValueError("--n_embd must be divisible by --n_head")

    print("[DEBUG] Starting the main training script.")
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    print(f"[DEBUG] Ensuring dataset '{dataset}' is available and loading data.")
    data_dir = ensure_dataset(dataset, prepare_data=prepare_data)
    train_data, val_data, meta = load_dataset(data_dir, default_vocab_size=vocab_size)
    effective_vocab_size = int(meta.get("vocab_size", vocab_size))

    print(f"[DEBUG] Creating output directory: {out_dir}")
    output_dir = ROOT_DIR / out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        saved_config = build_config_dict()
        saved_config["vocab_size"] = effective_vocab_size
        json.dump(saved_config, f, indent=2)

    print("[DEBUG] Initializing the GPT2 model architecture.")
    model = GPT2(
        vocab_size=effective_vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        n_positions=block_size,
    )
    print("[DEBUG] Collecting parameters for the optimizer.")
    parameters = collect_parameters(model)
    optimizer = AdamW(
        parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    loss_fn = CrossEntropyLoss()

    param_count = sum(p.data.size for p in parameters)
    print(f"dataset: {display_path(data_dir)}")
    print(f"vocab size: {effective_vocab_size}")
    print(f"model parameters: {param_count:,}")
    print(f"tokens per iteration: {gradient_accumulation_steps * batch_size * block_size:,}")

    # [ADDED] History tracker for Matplotlib
    history = {"iter": [], "train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": []}

    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=build_config_dict())

    best_val_loss = float("inf")
    t0 = time.time()

    print(f"[DEBUG] Entering the training loop for {max_iters} iterations.")
    for iter_num in range(max_iters + 1):
        lr = (
            learning_rate
            if not decay_lr
            else get_lr(iter_num, learning_rate, warmup_iters, lr_decay_iters, min_lr)
        )
        optimizer.set_lr(lr)

        if iter_num % eval_interval == 0:
            metrics = estimate_loss(
                model,
                train_data,
                val_data,
                batch_size,
                block_size,
                eval_iters,
                rng,
            )
            
            # [ADDED] Store history and print enhanced metrics
            history["iter"].append(iter_num)
            history["train_loss"].append(metrics['train']['loss'])
            history["val_loss"].append(metrics['val']['loss'])
            history["train_ppl"].append(metrics['train']['perplexity'])
            history["val_ppl"].append(metrics['val']['perplexity'])

            print(
                f"step {iter_num}: "
                f"train loss {metrics['train']['loss']:.4f} (PPL: {metrics['train']['perplexity']:.2f}, BPC: {metrics['train']['bpc']:.2f}) | "
                f"val loss {metrics['val']['loss']:.4f} (PPL: {metrics['val']['perplexity']:.2f}, BPC: {metrics['val']['bpc']:.2f})"
            )
            
            # [ADDED] Generate/Update Plot Artifacts
            plot_metrics(history, output_dir)
            
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": metrics["train"]["loss"],
                    "val/loss": metrics["val"]["loss"],
                    "train/perplexity": metrics["train"]["perplexity"],
                    "val/perplexity": metrics["val"]["perplexity"],
                    "lr": lr,
                })
                
            if metrics["val"]["loss"] < best_val_loss or always_save_checkpoint:
                best_val_loss = metrics["val"]["loss"]
                save_checkpoint(output_dir / "ckpt_best_v3.npz", parameters, iter_num, best_val_loss)
                print(f"saved best checkpoint to {display_path(output_dir / 'ckpt_best_v3.npz')}")

        if iter_num == 0 and eval_only:
            break

        accumulated_grads = [np.zeros_like(p.data) for p in parameters]
        loss_total = 0.0
        for _ in range(gradient_accumulation_steps):
            x, y = get_batch(train_data, batch_size, block_size, rng)
            optimizer.zero_grad()
            logits = model.forward(x)
            loss = loss_fn.forward(logits, y)
            grad_logits = loss_fn.backward()
            model.backward(grad_logits)
            loss_total += loss / gradient_accumulation_steps
            for param_idx, param in enumerate(parameters):
                grad = param.grad
                if grad is not None:
                    accumulated_grads[param_idx] += grad / gradient_accumulation_steps

        for param, grad in zip(parameters, accumulated_grads):
            setattr(param.owner, param.grad_name, grad)

        grad_norm = clip_grad_norm(parameters, grad_clip)
        optimizer.step()

        if iter_num % log_interval == 0:
            t1 = time.time()
            dt_ms = (t1 - t0) * 1000.0
            t0 = t1
            print(
                f"iter {iter_num}: loss {loss_total:.4f}, lr {lr:.2e}, "
                f"grad_norm {grad_norm:.4f}, time {dt_ms:.2f}ms"
            )
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/iter_loss": loss_total,
                    "grad_norm": grad_norm,
                    "dt_ms": dt_ms,
                    "lr": lr,
                })

    save_checkpoint(output_dir / "ckpt_last_v3.npz", parameters, max_iters, best_val_loss)
    print(f"saved last checkpoint to {display_path(output_dir / 'ckpt_last_v3.npz')}")
    
    # [ADDED] Run the Downstream Evaluation phase at the end of training
    if not eval_only:
        evaluate_downstream(model, meta, block_size, batch_size, rng)

    if sample_tokens > 0:
        print("\n--- sample ---")
        if "stoi" in meta and "itos" in meta:
            print(
                sample_text(
                    model,
                    meta,
                    sample_prompt,
                    block_size,
                    sample_tokens,
                    temperature,
                    rng,
                )
            )
        else:
            print("sampling skipped because this dataset has no character decoder in meta.pkl")


if __name__ == "__main__":
    main() 