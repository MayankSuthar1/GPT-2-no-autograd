import os
import math
import csv
import json
import time
import modal
import numpy as np
from pathlib import Path

# --- Modal Setup ---
APP_NAME = "nanogpt-benchmark-suite"
VOLUME_NAME = "nanogpt-no-cupy-autograd-runs"
volume = modal.Volume.from_name(VOLUME_NAME)
REMOTE_ROOT = Path("/app")

# FIXED: Pinned datasets to 3.6.0 to bypass the 4.0.0 script deprecation crash
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy", "cupy-cuda12x", "tiktoken", "datasets==3.6.0", "tqdm", "matplotlib")
    .workdir(REMOTE_ROOT.as_posix())
    .add_local_dir(
        ".",
        REMOTE_ROOT.as_posix(),
        ignore=[".git", ".vscode", "__pycache__", "*.pyc", "out-*"]
    )
)
app = modal.App(APP_NAME)


def calculate_ece(confidences, accuracies, n_bins=10):
    """Calculates Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        in_bin = [j for j, conf in enumerate(confidences) if bin_lower <= conf <= bin_upper]
        if len(in_bin) > 0:
            bin_acc = np.mean([accuracies[j] for j in in_bin])
            bin_conf = np.mean([confidences[j] for j in in_bin])
            bin_weight = len(in_bin) / len(confidences)
            ece += bin_weight * abs(bin_acc - bin_conf)
    return ece


def save_benchmark_artifacts(results, out_dir):
    """Generates and saves the tables and a 2x2 dashboard for the entire benchmark suite."""
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    print("\n--- Saving Benchmark Artifacts ---")
    
    # 1. Save Tabular Data (CSV)
    csv_path = out_dir / "full_benchmark_results.csv"
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Benchmark/Metric", "Score", "Unit"])
        writer.writerow(["WikiText-103 Loss", f"{results['wikitext_loss']:.4f}", "Cross Entropy"])
        writer.writerow(["WikiText-103 Perplexity", f"{results['wikitext_perplexity']:.2f}", "PPL"])
        
        # Accuracies
        writer.writerow(["HellaSwag Accuracy", f"{results['hellaswag_acc']:.2f}", "%"])
        writer.writerow(["PIQA Accuracy", f"{results['piqa_acc']:.2f}", "%"])
        writer.writerow(["WinoGrande Accuracy", f"{results['winogrande_acc']:.2f}", "%"])
        writer.writerow(["OpenBookQA Accuracy", f"{results['obqa_acc']:.2f}", "%"])
        
        # Calibration (ECE)
        writer.writerow(["Average Calibration Error (ECE)", f"{results['avg_ece']:.4f}", "Rate (0-1)"])
        
        # Generation Speed
        writer.writerow(["Time To First Token (TTFT)", f"{results['ttft_ms']:.2f}", "ms"])
        writer.writerow(["Generation Speed (TPS)", f"{results['tps']:.2f}", "tokens/sec"])
    print(f"Saved Table: {csv_path}")

    # 2. Save JSON Metadata 
    json_path = out_dir / "full_benchmark_metadata.json"
    with open(json_path, "w") as file:
        json.dump(results, file, indent=4)
    
    # 3. Generate 2x2 Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Top-Left: Perplexity
    axs[0, 0].bar(["WikiText-103"], [results["wikitext_perplexity"]], color='skyblue', edgecolor='black', width=0.4)
    axs[0, 0].set_title("Language Modeling (Perplexity)\n< Lower is Better", fontweight='bold')
    axs[0, 0].text(0, results["wikitext_perplexity"] + 1, f"{results['wikitext_perplexity']:.2f}", ha='center', fontweight='bold')

    # Top-Right: Accuracies
    tasks = ["HellaSwag", "PIQA", "WinoGrande", "OBQA"]
    accs = [results["hellaswag_acc"], results["piqa_acc"], results["winogrande_acc"], results["obqa_acc"]]
    colors = ['lightgreen', 'lightcoral', 'gold', 'plum']
    bars = axs[0, 1].bar(tasks, accs, color=colors, edgecolor='black')
    axs[0, 1].set_title("Zero-Shot Accuracies\nHigher is Better >", fontweight='bold')
    axs[0, 1].set_ylim(0, 100)
    axs[0, 1].axhline(50, color='gray', linestyle='--', label="2-Choice Baseline")
    axs[0, 1].axhline(25, color='red', linestyle='--', label="4-Choice Baseline")
    axs[0, 1].legend()
    for i, bar in enumerate(bars):
        axs[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{accs[i]:.1f}%", ha='center')

    # Bottom-Left: Calibration Error (ECE)
    axs[1, 0].bar(["Avg Expected Calibration Error"], [results["avg_ece"]], color='orange', edgecolor='black', width=0.4)
    axs[1, 0].set_title("Model Confidence Calibration (ECE)\n< Lower is Better", fontweight='bold')
    axs[1, 0].set_ylim(0, 1.0)
    axs[1, 0].text(0, results["avg_ece"] + 0.05, f"{results['avg_ece']:.4f}", ha='center', fontweight='bold')

    # Bottom-Right: TPS & TTFT (FIXED MATPLOTLIB SYNTAX)
    ax3 = axs[1, 1]
    ax4 = ax3.twinx()
    
    # Use explicit X-coordinates (0 and 1) to place them side-by-side
    ax3.bar([0], [results["ttft_ms"]], color='teal', edgecolor='black', width=0.4)
    ax4.bar([1], [results["tps"]], color='purple', edgecolor='black', width=0.4)
    
    # Set the labels for those coordinates manually
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Time To First Token", "Tokens Per Second"])
    
    ax3.set_ylabel("TTFT (ms)", color='teal', fontweight='bold')
    ax4.set_ylabel("Speed (Tokens/sec)", color='purple', fontweight='bold')
    axs[1, 1].set_title("Inference Performance (A100)\nSpeed vs Latency", fontweight='bold')
    
    plt.tight_layout()
    graph_path = out_dir / "benchmark_dashboard.png"
    plt.savefig(graph_path, dpi=300)
    plt.close()
    print(f"Saved Dashboard: {graph_path}")

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume},
    timeout=60 * 90, 
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_benchmarks():
    import sys
    import cupy as cp
    import numpy as np
    from tqdm import tqdm
    import tiktoken
    from datasets import load_dataset
    
    sys.path.insert(0, str(REMOTE_ROOT / "no_autograd"))
    from gpt_ffn_bp_v3_cupy import GPT2, CrossEntropyLoss
    from train_gpt_ffn_bp_v3_cupy import collect_parameters

    print("\n--- Initializing Custom CuPy Model ---")
    block_size = 1024
    model = GPT2(vocab_size=50304, n_embd=128, n_head=4, n_layer=10, n_positions=block_size)
    parameters = collect_parameters(model)
    
    out_dir = "/checkpoints/out-gpt-ffn-bp-v3-cupy-openwebtext"
    ckpt_path = f"{out_dir}/ckpt_best_v3_cupy.npz"
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Could not find checkpoint at {ckpt_path}")
        return

    print(f"Loading weights from {ckpt_path}...")
    npz_file = np.load(ckpt_path)
    for p in parameters:
        if p.key in npz_file:
            p.data[...] = cp.asarray(npz_file[p.key])
    
    enc = tiktoken.get_encoding("gpt2")
    results = {}

    def get_sequence_loss(seq_tokens):
        if len(seq_tokens) > block_size + 1:
            seq_tokens = seq_tokens[-(block_size + 1):]
        x = cp.array([seq_tokens[:-1]], dtype=cp.int64)
        y = cp.array([seq_tokens[1:]], dtype=cp.int64)
        
        logits = model.forward(x)
        shifted = logits - cp.max(logits, axis=-1, keepdims=True)
        probs = cp.exp(shifted)
        probs = probs / cp.sum(probs, axis=-1, keepdims=True)
        
        correct_probs = probs[0, cp.arange(y.shape[1]), y[0]]
        return -float(cp.mean(cp.log(correct_probs + 1e-10)))

    def eval_zero_shot(dataset, task_name, extract_fn):
        correct = 0
        confidences = []
        accuracies = []
        
        for example in tqdm(dataset, desc=task_name):
            choices_seqs, label = extract_fn(example)
            
            choice_losses = [get_sequence_loss(seq) for seq in choices_seqs]
            predicted_idx = np.argmin(choice_losses)
            
            shifted_losses = [-l - max([-x for x in choice_losses]) for l in choice_losses]
            exp_vals = np.exp(shifted_losses)
            probs = exp_vals / np.sum(exp_vals)
            
            confidences.append(probs[predicted_idx])
            is_correct = 1 if predicted_idx == label else 0
            accuracies.append(is_correct)
            correct += is_correct
            
        acc = (correct / len(dataset)) * 100
        ece = calculate_ece(confidences, accuracies)
        return acc, ece

    # ==========================================
    # 1. WikiText-103 (Perplexity)
    # ==========================================
    print("--- [1/6] Evaluating WikiText-103 ---")
    
    # FIXED: Added trust_remote_code=True
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test", trust_remote_code=True)
    full_text = "\n\n".join(dataset["text"])
    tokens = enc.encode_ordinary(full_text)
    
    batch_size = 16 
    inputs, targets = [], []
    for i in range(0, len(tokens) - block_size, block_size):
        inputs.append(tokens[i : i + block_size])
        targets.append(tokens[i + 1 : i + 1 + block_size])
    
    total_loss, total_batches = 0.0, 0
    loss_fn = CrossEntropyLoss()
    
    for i in tqdm(range(0, len(inputs), batch_size), desc="WikiText-103"):
        batch_x = inputs[i : i + batch_size]
        batch_y = targets[i : i + batch_size]
        if len(batch_x) < batch_size: continue
        
        x = cp.array(batch_x, dtype=cp.int64)
        y = cp.array(batch_y, dtype=cp.int64)
        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)
        total_loss += float(loss)
        total_batches += 1
        
    results["wikitext_loss"] = total_loss / total_batches
    results["wikitext_perplexity"] = math.exp(results["wikitext_loss"])
    print(f"-> PPL: {results['wikitext_perplexity']:.2f}\n")

    # ==========================================
    # 2-5. Zero-Shot Benchmarks (Accuracy & Calibration)
    # ==========================================
    print("--- [2-5/6] Evaluating Zero-Shot Accuracies & Calibration ---")
    
    def hs_extract(ex):
        ctx = enc.encode_ordinary(ex['ctx'])
        return [ctx + enc.encode_ordinary(" " + end) for end in ex['endings']], int(ex['label'])
        
    def piqa_extract(ex):
        return [enc.encode_ordinary(ex['goal'] + " " + ans) for ans in [ex['sol1'], ex['sol2']]], int(ex['label'])
        
    def wino_extract(ex):
        return [enc.encode_ordinary(ex['sentence'].replace("_", opt)) for opt in [ex['option1'], ex['option2']]], int(ex['answer']) - 1
        
    def obqa_extract(ex):
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        return [enc.encode_ordinary(ex['question_stem'] + " " + c) for c in ex['choices']['text']], label_map[ex['answerKey']]

    # FIXED: Added trust_remote_code=True to every call
    results["hellaswag_acc"], hs_ece = eval_zero_shot(
        load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True), "HellaSwag", hs_extract)
        
    results["piqa_acc"], piqa_ece = eval_zero_shot(
        load_dataset("ybisk/piqa", split="validation", trust_remote_code=True), "PIQA", piqa_extract)
        
    results["winogrande_acc"], wino_ece = eval_zero_shot(
        load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True), "WinoGrande", wino_extract)
        
    results["obqa_acc"], obqa_ece = eval_zero_shot(
        load_dataset("allenai/openbookqa", "main", split="validation", trust_remote_code=True), "OpenBookQA", obqa_extract)
    
    results["avg_ece"] = np.mean([hs_ece, piqa_ece, wino_ece, obqa_ece])
    print(f"-> Average Calibration Error (ECE): {results['avg_ece']:.4f}\n")

    # ==========================================
    # 6. TTFT and TPS (Performance Profiling)
    # ==========================================
    print("--- [6/6] Profiling Generation Speed (TTFT & TPS) ---")
    prompt = "The future of artificial intelligence requires highly optimized"
    input_ids = enc.encode_ordinary(prompt)
    x = cp.array([input_ids], dtype=cp.int64)
    
    # Warmup kernel compilation
    _ = model.forward(x)
    
    # TTFT: Time to process prompt and get first token
    start_ttft = time.time()
    logits = model.forward(x)
    results["ttft_ms"] = (time.time() - start_ttft) * 1000.0
    
    # TPS: Autoregressive decoding speed
    gen_tokens = 50
    start_tps = time.time()
    for _ in range(gen_tokens):
        next_id = int(cp.argmax(logits[0, -1, :]))
        x = cp.append(x, cp.array([[next_id]], dtype=cp.int64), axis=1)
        if x.shape[1] > block_size:
            x = x[:, -block_size:]
        logits = model.forward(x)
        
    total_time_sec = time.time() - start_tps
    results["tps"] = gen_tokens / total_time_sec
    
    print(f"-> TTFT: {results['ttft_ms']:.2f} ms")
    print(f"-> TPS:  {results['tps']:.2f} tokens/sec\n")

    # ==========================================
    # 7. Sync Volume
    # ==========================================
    save_benchmark_artifacts(results, out_dir)
    print("Committing artifacts to the persistent volume...")
    volume.commit()
    print("All benchmarks complete!")

@app.local_entrypoint()
def main():
    run_benchmarks.remote()