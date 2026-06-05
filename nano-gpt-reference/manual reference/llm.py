import glob
import os
import sys
import time

import numpy as np


PARAMETER_NAMES = [
    "wte", "wpe", "ln1w", "ln1b", "qkvw", "qkvb", "attprojw", "attprojb",
    "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb", "lnfw", "lnfb",
]

ACTIVATION_NAMES = [
    "encoded", "ln1", "ln1_mean", "ln1_rstd", "qkv", "atty", "preatt", "att",
    "attproj", "residual2", "ln2", "ln2_mean", "ln2_rstd", "fch", "fch_gelu",
    "fcproj", "residual3", "lnf", "lnf_mean", "lnf_rstd", "logits", "probs",
    "losses",
]


class Obj:
    pass


class GPT2Config:
    def __init__(self, max_seq_len=0, vocab_size=0, padded_vocab_size=0,
                 num_layers=0, num_heads=0, channels=0):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.channels = channels


def make_views(names, shapes, memory):
    out = Obj()
    offset = 0
    for name, shape in zip(names, shapes):
        size = int(np.prod(shape))
        setattr(out, name, memory[offset:offset + size].reshape(shape))
        offset += size
    return out


def parameter_shapes(config):
    Vp = config.padded_vocab_size
    C = config.channels
    maxT = config.max_seq_len
    L = config.num_layers
    return [
        (Vp, C),
        (maxT, C),
        (L, C),
        (L, C),
        (L, 3 * C, C),
        (L, 3 * C),
        (L, C, C),
        (L, C),
        (L, C),
        (L, C),
        (L, 4 * C, C),
        (L, 4 * C),
        (L, C, 4 * C),
        (L, C),
        (C,),
        (C,),
    ]


def activation_shapes(config, B, T):
    C = config.channels
    NH = config.num_heads
    L = config.num_layers
    Vp = config.padded_vocab_size
    return [
        (B, T, C),
        (L, B, T, C),
        (L, B, T),
        (L, B, T),
        (L, B, T, 3 * C),
        (L, B, T, C),
        (L, B, NH, T, T),
        (L, B, NH, T, T),
        (L, B, T, C),
        (L, B, T, C),
        (L, B, T, C),
        (L, B, T),
        (L, B, T),
        (L, B, T, 4 * C),
        (L, B, T, 4 * C),
        (L, B, T, C),
        (L, B, T, C),
        (B, T, C),
        (B, T),
        (B, T),
        (B, T, Vp),
        (B, T, Vp),
        (B, T),
    ]


def encoder_forward(inp, wte, wpe):
    B, T = inp.shape
    return wte[inp] + wpe[np.arange(T)][None, :, :]


def encoder_backward(dwte, dwpe, dout, inp):
    np.add.at(dwte, inp.reshape(-1), dout.reshape(-1, dout.shape[-1]))
    dwpe[:inp.shape[1]] += dout.sum(axis=0)


def layernorm_forward(inp, weight, bias):
    mean = inp.mean(axis=-1)
    xshift = inp - mean[..., None]
    var = (xshift * xshift).mean(axis=-1)
    rstd = 1.0 / np.sqrt(var + 1e-5)
    out = xshift * rstd[..., None]
    out = out * weight + bias
    return out.astype(np.float32), mean.astype(np.float32), rstd.astype(np.float32)


def layernorm_backward(dinp, dweight, dbias, dout, inp, weight, mean, rstd):
    norm = (inp - mean[..., None]) * rstd[..., None]
    dnorm = dout * weight
    dnorm_mean = dnorm.mean(axis=-1, keepdims=True)
    dnorm_norm_mean = (dnorm * norm).mean(axis=-1, keepdims=True)
    dinp += (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd[..., None]
    dweight += (norm * dout).sum(axis=(0, 1))
    dbias += dout.sum(axis=(0, 1))


def matmul_forward(inp, weight, bias):
    out = inp @ weight.T
    if bias is not None:
        out = out + bias
    return out.astype(np.float32)


def matmul_backward(dinp, dweight, dbias, dout, inp, weight):
    B, T, C = inp.shape
    OC = dout.shape[-1]
    dout2 = dout.reshape(B * T, OC)
    inp2 = inp.reshape(B * T, C)
    dinp += (dout2 @ weight).reshape(B, T, C)
    dweight += dout2.T @ inp2
    if dbias is not None:
        dbias += dout2.sum(axis=0)


def attention_forward(inp, B, T, C, NH):
    hs = C // NH
    qkv = inp.reshape(B, T, 3, NH, hs)
    q = qkv[:, :, 0].transpose(0, 2, 1, 3)
    k = qkv[:, :, 1].transpose(0, 2, 1, 3)
    v = qkv[:, :, 2].transpose(0, 2, 1, 3)
    scale = 1.0 / np.sqrt(hs)

    preatt = (q @ k.transpose(0, 1, 3, 2)) * scale
    mask = np.tril(np.ones((T, T), dtype=bool))[None, None, :, :]
    masked = np.where(mask, preatt, -1e10)
    shifted = masked - masked.max(axis=-1, keepdims=True)
    expv = np.exp(shifted).astype(np.float32)
    att = expv / expv.sum(axis=-1, keepdims=True)
    att = np.where(mask, att, 0.0).astype(np.float32)

    out = att @ v
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    return out.astype(np.float32), masked.astype(np.float32), att


def attention_backward(dinp, dpreatt, datt, dout, inp, att, B, T, C, NH):
    hs = C // NH
    qkv = inp.reshape(B, T, 3, NH, hs)
    q = qkv[:, :, 0].transpose(0, 2, 1, 3)
    k = qkv[:, :, 1].transpose(0, 2, 1, 3)
    v = qkv[:, :, 2].transpose(0, 2, 1, 3)
    dout_heads = dout.reshape(B, T, NH, hs).transpose(0, 2, 1, 3)
    scale = 1.0 / np.sqrt(hs)

    local_datt = dout_heads @ v.transpose(0, 1, 3, 2)
    local_dv = att.transpose(0, 1, 3, 2) @ dout_heads
    local_dpreatt = att * (local_datt - (local_datt * att).sum(axis=-1, keepdims=True))
    mask = np.tril(np.ones((T, T), dtype=bool))[None, None, :, :]
    local_datt = np.where(mask, local_datt, 0.0)
    local_dpreatt = np.where(mask, local_dpreatt, 0.0)

    local_dq = (local_dpreatt @ k) * scale
    local_dk = (local_dpreatt.transpose(0, 1, 3, 2) @ q) * scale

    datt += local_datt.astype(np.float32)
    dpreatt += local_dpreatt.astype(np.float32)
    dinp_view = dinp.reshape(B, T, 3, NH, hs)
    dinp_view[:, :, 0] += local_dq.transpose(0, 2, 1, 3)
    dinp_view[:, :, 1] += local_dk.transpose(0, 2, 1, 3)
    dinp_view[:, :, 2] += local_dv.transpose(0, 2, 1, 3)


def gelu_forward(inp):
    x = inp
    cube = 0.044715 * x * x * x
    return (0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + cube)))).astype(np.float32)


def gelu_backward(dinp, inp, dout):
    x = inp
    cube = 0.044715 * x * x * x
    tanh_arg = np.sqrt(2.0 / np.pi) * (x + cube)
    tanh_out = np.tanh(tanh_arg)
    sech_out = 1.0 / (np.cosh(tanh_arg) ** 2)
    local_grad = 0.5 * (1.0 + tanh_out)
    local_grad += x * 0.5 * sech_out * np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x * x)
    dinp += local_grad * dout


def residual_forward(inp1, inp2):
    return (inp1 + inp2).astype(np.float32)


def residual_backward(dinp1, dinp2, dout):
    dinp1 += dout
    dinp2 += dout


def softmax_forward(logits, V, Vp):
    probs = np.zeros_like(logits)
    useful = logits[:, :, :V]
    shifted = useful - useful.max(axis=-1, keepdims=True)
    expv = np.exp(shifted).astype(np.float32)
    probs[:, :, :V] = expv / expv.sum(axis=-1, keepdims=True)
    if V < Vp:
        probs[:, :, V:Vp] = 0.0
    return probs


def crossentropy_forward(probs, targets):
    B, T = targets.shape
    b = np.arange(B)[:, None]
    t = np.arange(T)[None, :]
    return (-np.log(probs[b, t, targets])).astype(np.float32)


def crossentropy_softmax_backward(dlogits, dlosses, probs, targets, V):
    dlogits[:, :, :V] += probs[:, :, :V] * dlosses[:, :, None] # what it is doing ? > 
    flat = dlogits[:, :, :V].reshape(-1, V)
    flat_targets = targets.reshape(-1)
    flat_losses = dlosses.reshape(-1)
    flat[np.arange(flat.shape[0]), flat_targets] -= flat_losses


class GPT2:
    def __init__(self):
        self.config = GPT2Config()
        self.params = None
        self.param_shapes = None
        self.params_memory = None
        self.num_parameters = 0
        self.grads = None
        self.grads_memory = None
        self.m_memory = None
        self.v_memory = None
        self.acts = None
        self.act_shapes = None
        self.acts_memory = None
        self.num_activations = 0
        self.grads_acts = None
        self.grads_acts_memory = None
        self.batch_size = 0
        self.seq_len = 0
        self.inputs = None
        self.targets = None
        self.mean_loss = -1.0


def gpt2_build_from_checkpoint(checkpoint_path):
    model = GPT2()
    with open(checkpoint_path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=256)
        if header[0] != 20240326:
            raise ValueError("bad magic model file")
        if header[1] != 3:
            raise ValueError("bad version in model file")

        model.config = GPT2Config(
            max_seq_len=int(header[2]),
            vocab_size=int(header[3]),
            num_layers=int(header[4]),
            num_heads=int(header[5]),
            channels=int(header[6]),
            padded_vocab_size=int(header[7]),
        )

        c = model.config
        print("[GPT-2]")
        print("max_seq_len:", c.max_seq_len)
        print("vocab_size:", c.vocab_size)
        print("padded_vocab_size:", c.padded_vocab_size)
        print("num_layers:", c.num_layers)
        print("num_heads:", c.num_heads)
        print("channels:", c.channels)

        model.param_shapes = parameter_shapes(model.config)
        model.num_parameters = sum(int(np.prod(s)) for s in model.param_shapes)
        print("num_parameters:", model.num_parameters)

        params = np.fromfile(f, dtype=np.float32, count=model.num_parameters)
        if params.size != model.num_parameters:
            raise ValueError("checkpoint ended before all parameters were read")
        model.params_memory = params
        model.params = make_views(PARAMETER_NAMES, model.param_shapes, model.params_memory)
    return model


def gpt2_forward(model, inputs, targets=None, B=None, T=None):
    inputs = np.asarray(inputs, dtype=np.int32)
    if inputs.ndim == 1:
        if B is None or T is None:
            raise ValueError("flat inputs need B and T")
        inputs = inputs.reshape(B, T)
    B, T = inputs.shape

    if targets is not None:
        targets = np.asarray(targets, dtype=np.int32)
        if targets.ndim == 1:
            targets = targets.reshape(B, T)

    V = model.config.vocab_size
    Vp = model.config.padded_vocab_size
    L = model.config.num_layers
    NH = model.config.num_heads
    C = model.config.channels

    assert np.all((0 <= inputs) & (inputs < V))
    if targets is not None:
        assert np.all((0 <= targets) & (targets < V))

    if model.acts_memory is None:
        model.batch_size = B
        model.seq_len = T
        model.act_shapes = activation_shapes(model.config, B, T)
        model.num_activations = sum(int(np.prod(s)) for s in model.act_shapes)
        print("num_activations:", model.num_activations)
        model.acts_memory = np.empty(model.num_activations, dtype=np.float32)
        model.acts = make_views(ACTIVATION_NAMES, model.act_shapes, model.acts_memory)
    elif B != model.batch_size or T != model.seq_len:
        raise ValueError("forward B,T must match the first forward call")

    model.inputs = inputs.copy()
    model.targets = None if targets is None else targets.copy()
    p = model.params
    a = model.acts

    a.encoded[...] = encoder_forward(inputs, p.wte, p.wpe)
    for l in range(L):
        residual = a.encoded if l == 0 else a.residual3[l - 1]

        a.ln1[l], a.ln1_mean[l], a.ln1_rstd[l] = layernorm_forward(residual, p.ln1w[l], p.ln1b[l])
        a.qkv[l] = matmul_forward(a.ln1[l], p.qkvw[l], p.qkvb[l])
        a.atty[l], a.preatt[l], a.att[l] = attention_forward(a.qkv[l], B, T, C, NH)
        a.attproj[l] = matmul_forward(a.atty[l], p.attprojw[l], p.attprojb[l])
        a.residual2[l] = residual_forward(residual, a.attproj[l])
        a.ln2[l], a.ln2_mean[l], a.ln2_rstd[l] = layernorm_forward(a.residual2[l], p.ln2w[l], p.ln2b[l])
        a.fch[l] = matmul_forward(a.ln2[l], p.fcw[l], p.fcb[l])
        a.fch_gelu[l] = gelu_forward(a.fch[l])
        a.fcproj[l] = matmul_forward(a.fch_gelu[l], p.fcprojw[l], p.fcprojb[l])
        a.residual3[l] = residual_forward(a.residual2[l], a.fcproj[l])

    residual = a.residual3[L - 1] if L > 0 else a.encoded
    a.lnf, a.lnf_mean, a.lnf_rstd = layernorm_forward(residual, p.lnfw, p.lnfb)
    a.logits[...] = matmul_forward(a.lnf, p.wte, None)
    a.probs[...] = softmax_forward(a.logits, V, Vp)

    if targets is not None:
        a.losses[...] = crossentropy_forward(a.probs, targets)
        model.mean_loss = float(a.losses.mean())
    else:
        model.mean_loss = -1.0
    return a.logits, model.mean_loss


def gpt2_zero_grad(model):
    if model.grads_memory is not None:
        model.grads_memory.fill(0.0)
    if model.grads_acts_memory is not None:
        model.grads_acts_memory.fill(0.0)


def gpt2_backward(model):
    if model.mean_loss == -1.0:
        raise ValueError("must forward with targets before backward")

    if model.grads_memory is None:
        model.grads_memory = np.zeros(model.num_parameters, dtype=np.float32)
        model.grads = make_views(PARAMETER_NAMES, model.param_shapes, model.grads_memory)
        model.grads_acts_memory = np.zeros(model.num_activations, dtype=np.float32)
        model.grads_acts = make_views(ACTIVATION_NAMES, model.act_shapes, model.grads_acts_memory)

    B = model.batch_size
    T = model.seq_len
    V = model.config.vocab_size
    Vp = model.config.padded_vocab_size
    L = model.config.num_layers
    NH = model.config.num_heads
    C = model.config.channels
    p = model.params
    g = model.grads
    a = model.acts
    ga = model.grads_acts

    ga.losses[...] = 1.0 / (B * T)
    crossentropy_softmax_backward(ga.logits, ga.losses, a.probs, model.targets, V)
    matmul_backward(ga.lnf, g.wte, None, ga.logits, a.lnf, p.wte)

    residual = a.residual3[L - 1] if L > 0 else a.encoded
    dresidual = ga.residual3[L - 1] if L > 0 else ga.encoded
    layernorm_backward(dresidual, g.lnfw, g.lnfb, ga.lnf, residual, p.lnfw, a.lnf_mean, a.lnf_rstd)

    for l in range(L - 1, -1, -1):
        residual = a.encoded if l == 0 else a.residual3[l - 1]
        dresidual = ga.encoded if l == 0 else ga.residual3[l - 1]

        residual_backward(ga.residual2[l], ga.fcproj[l], ga.residual3[l])
        matmul_backward(ga.fch_gelu[l], g.fcprojw[l], g.fcprojb[l],
                        ga.fcproj[l], a.fch_gelu[l], p.fcprojw[l])
        gelu_backward(ga.fch[l], a.fch[l], ga.fch_gelu[l])
        matmul_backward(ga.ln2[l], g.fcw[l], g.fcb[l], ga.fch[l], a.ln2[l], p.fcw[l])
        layernorm_backward(ga.residual2[l], g.ln2w[l], g.ln2b[l],
                           ga.ln2[l], a.residual2[l], p.ln2w[l], a.ln2_mean[l], a.ln2_rstd[l])
        residual_backward(dresidual, ga.attproj[l], ga.residual2[l])
        matmul_backward(ga.atty[l], g.attprojw[l], g.attprojb[l],
                        ga.attproj[l], a.atty[l], p.attprojw[l])
        attention_backward(ga.qkv[l], ga.preatt[l], ga.att[l],
                           ga.atty[l], a.qkv[l], a.att[l], B, T, C, NH)
        matmul_backward(ga.ln1[l], g.qkvw[l], g.qkvb[l], ga.qkv[l], a.ln1[l], p.qkvw[l])
        layernorm_backward(dresidual, g.ln1w[l], g.ln1b[l],
                           ga.ln1[l], residual, p.ln1w[l], a.ln1_mean[l], a.ln1_rstd[l])

    encoder_backward(g.wte, g.wpe, ga.encoded, model.inputs)


def gpt2_update(model, learning_rate, beta1, beta2, eps, weight_decay, t):
    if model.m_memory is None:
        model.m_memory = np.zeros_like(model.params_memory)
        model.v_memory = np.zeros_like(model.params_memory)

    param = model.params_memory
    grad = model.grads_memory
    model.m_memory[...] = beta1 * model.m_memory + (1.0 - beta1) * grad
    model.v_memory[...] = beta2 * model.v_memory + (1.0 - beta2) * grad * grad
    m_hat = model.m_memory / (1.0 - beta1 ** t)
    v_hat = model.v_memory / (1.0 - beta2 ** t)
    param -= learning_rate * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * param)


def random_u32(state):
    state[0] ^= state[0] >> 12
    state[0] ^= (state[0] << 25) & 0xFFFFFFFFFFFFFFFF
    state[0] ^= state[0] >> 27
    state[0] &= 0xFFFFFFFFFFFFFFFF
    return ((state[0] * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF) >> 32


def random_f32(state):
    return (random_u32(state) >> 8) / 16777216.0


def sample_mult(probabilities, n, coin):
    cdf = np.cumsum(probabilities[:n])
    ix = int(np.searchsorted(cdf, coin, side="right"))
    return min(ix, n - 1)


class Tokenizer:
    def __init__(self, filename):
        self.init_ok = False
        self.vocab_size = 0
        self.eot_token = 50256
        self.token_table = []
        if not os.path.exists(filename):
            print("---")
            print("WARNING: Failed to open the tokenizer file", filename)
            print("---")
            return
        with open(filename, "rb") as f:
            header = np.fromfile(f, dtype=np.uint32, count=256)
            if header[0] != 20240328:
                raise ValueError("bad magic tokenizer file")
            version = int(header[1])
            self.vocab_size = int(header[2])
            if version == 1:
                self.eot_token = 50256
            elif version == 2:
                self.eot_token = int(header[3])
            else:
                raise ValueError("bad tokenizer version")
            for _ in range(self.vocab_size):
                length = f.read(1)[0]
                self.token_table.append(f.read(length))
        self.init_ok = True

    def decode(self, token_id):
        if not self.init_ok or token_id >= self.vocab_size:
            return None
        return self.token_table[token_id]


def safe_print(piece):
    if piece is None or len(piece) == 0:
        return
    if len(piece) == 1:
        b = piece[0]
        if not (32 <= b <= 126 or b in (9, 10, 13)):
            return
    sys.stdout.buffer.write(piece)
    sys.stdout.flush()


class DataLoader:
    def __init__(self, filename_pattern, B, T, process_rank=0, num_processes=1, should_shuffle=0):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.should_shuffle = bool(should_shuffle)
        self.files = sorted(glob.glob(filename_pattern))
        if len(self.files) == 0:
            raise FileNotFoundError("no files found matching " + filename_pattern)
        self.rng = np.random.default_rng(42 + process_rank)
        self.shard_indices = np.arange(len(self.files))
        self.current_shard_idx = 0
        self.current_sample_idx = 0
        self.tokens = None
        self.num_tokens = 0
        for name in self.files:
            self.num_tokens += self._read_token_count(name)
        self.inputs = None
        self.targets = None
        self.reset()

    def _read_token_count(self, filename):
        with open(filename, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=256)
        if header[0] == 20240520 and header[1] == 1:
            return int(header[2])
        size = os.path.getsize(filename)
        return size // np.dtype(np.uint16).itemsize

    def _read_tokens(self, filename):
        with open(filename, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=256)
            if header[0] == 20240520 and header[1] == 1:
                ntok = int(header[2])
                tokens = np.fromfile(f, dtype=np.uint16, count=ntok)
            else:
                f.seek(0)
                tokens = np.fromfile(f, dtype=np.uint16)
        return tokens.astype(np.int32)

    def _load_shard(self, shard_index):
        real_index = self.shard_indices[shard_index] if self.should_shuffle else shard_index
        self.tokens = self._read_tokens(self.files[int(real_index)])
        total_batch = self.num_processes * self.B * self.T
        self.shard_num_samples = (len(self.tokens) - 1) // total_batch
        if self.shard_num_samples <= 0:
            raise ValueError("shard is too small for this B, T, and process count")
        if self.should_shuffle:
            self.intra_shard_indices = self.rng.permutation(self.shard_num_samples)
        else:
            self.intra_shard_indices = np.arange(self.shard_num_samples)

    def reset(self):
        self.current_shard_idx = 0
        self.current_sample_idx = 0
        if self.should_shuffle:
            self.rng.shuffle(self.shard_indices)
        self._load_shard(self.current_shard_idx)

    def _advance(self):
        self.current_shard_idx += 1
        if self.current_shard_idx >= len(self.files):
            self.reset()
        else:
            self.current_sample_idx = 0
            self._load_shard(self.current_shard_idx)

    def next_batch(self):
        if self.current_sample_idx >= self.shard_num_samples:
            self._advance()
        idx = int(self.intra_shard_indices[self.current_sample_idx])
        total_batch = self.num_processes * self.B * self.T
        start = idx * total_batch + self.process_rank * self.B * self.T
        buf = self.tokens[start:start + self.B * self.T + 1]
        self.inputs = buf[:-1].reshape(self.B, self.T).astype(np.int32)
        self.targets = buf[1:].reshape(self.B, self.T).astype(np.int32)
        self.current_sample_idx += 1
        return self.inputs, self.targets


def choose_data_files():
    tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin"
    tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin"
    tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin"
    tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin"
    train_tokens = tiny_shakespeare_train if os.path.exists(tiny_shakespeare_train) else tiny_stories_train
    val_tokens = tiny_shakespeare_val if os.path.exists(tiny_shakespeare_val) else tiny_stories_val
    return train_tokens, val_tokens


def main():
    model = gpt2_build_from_checkpoint("gpt2_124M.bin")
    train_tokens, val_tokens = choose_data_files()
    B = 4
    T = 64
    train_loader = DataLoader(train_tokens, B, T, 0, 1, 1)
    val_loader = DataLoader(val_tokens, B, T, 0, 1, 0)
    print("train dataset num_batches:", train_loader.num_tokens // (B * T))
    print("val dataset num_batches:", val_loader.num_tokens // (B * T))
    val_num_batches = 5

    tokenizer = Tokenizer("gpt2_tokenizer.bin")
    rng_state = [1337]
    gen_tokens = np.empty((B, T), dtype=np.int32)
    genT = 64

    for step in range(41):
        if step % 10 == 0:
            val_loss = 0.0
            val_loader.reset()
            for _ in range(val_num_batches):
                x, y = val_loader.next_batch()
                gpt2_forward(model, x, y)
                val_loss += model.mean_loss
            val_loss /= val_num_batches
            print("val loss %f" % val_loss)

        if step > 0 and step % 20 == 0:
            gen_tokens.fill(tokenizer.eot_token)
            print("generating:\n---")
            for t in range(1, genT):
                gpt2_forward(model, gen_tokens, None)
                probs = model.acts.probs[0, t - 1]
                coin = random_f32(rng_state)
                next_token = sample_mult(probs, model.config.vocab_size, coin)
                gen_tokens[0, t] = next_token
                if tokenizer.init_ok:
                    safe_print(tokenizer.decode(next_token))
                else:
                    print(next_token, end=" ")
            print("\n---")

        start = time.perf_counter()
        x, y = train_loader.next_batch()
        gpt2_forward(model, x, y)
        gpt2_zero_grad(model)
        gpt2_backward(model)
        gpt2_update(model, 1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print("step %d: train loss %f (took %f ms)" % (step, model.mean_loss, elapsed_ms))


if __name__ == "__main__":
    main()
