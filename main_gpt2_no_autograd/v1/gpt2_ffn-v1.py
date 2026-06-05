### V1 - simple gpt-2 transformer ffn using pre-trained weights from the released open-ai gpt-2 files.
## only understanding the feed forward network working and math of the transformer block

import numpy as np

def linear(x, w, b = None):
    output = np.dot(x, w)
    if b is not None:
        output += b
    return output

        
## softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis = -1, keepdims = True))
    return exp_x / np.sum(exp_x, axis = -1, keepdims = True)

## GELU activation function
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

## Layer normalization
def  layer_norm(x, g, b, eps = 1.e-5):
    mean = np.mean(x, axis = -1, keepdims = True)
    var = np.var(x, axis = -1, keepdims = True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return g * x_norm + b

## gpt-2 Feed forward network
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    # token + position embedding
    x = wte[inputs] + wpe [range(len(inputs))]
    
    # forward pass through n_layers of transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head = n_head)
        
    x = layer_norm(x, **ln_f)
    
    return x @ wte.T  # output logits for each token in the vocabulary

## gpt-2 transformer block
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    # multi-head self attention
    x = x + mh_c_s_a(layer_norm(x, **ln_1), **attn, n_head = n_head)
    
    # feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    
    return x

## block feed forward network
def ffn(x, c_fc, c_proj):
    #project up 
    a = gelu(linear(x, **c_fc))
    #project back down
    x = linear(a, **c_proj)
    return x

## normal attention function
def attention(q, k, v, mask):
    
    # compute attention scores with masking for causal attention
    attention_scores = q @ k.T / np.sqrt(q.shape[-1]) + mask
    
    # apply softmax to get attention weights
    attention_weights = softmax(attention_scores)
    
    # compute attention output
    output = attention_weights @ v
    return output

## self attention function without masking
def self_attention(x, c_attn, c_proj):
    
    # qkv projections
    x = linear(x, **c_attn)
    
    # split into q,k,v
    q,k,v = np.split(x, 3, axis = -1)
    
    
    # compute self attention output
    x = attention(q, k, v)
    
    # output projection
    x = linear(x, **c_proj)
    
    return x

## causal self attention function with masking to prevent attending to future tokens
def causal_self_attention(x, c_attn, c_proj):
    
    # qkv projections
    x = linear(x, **c_attn)
    
    # split into q,k,v
    q,k,v = np.split(x, 3, axis = -1)
    
    # mask values for causal attention (upper triangular mask)
    mask = (1- np.tri(x.shape[0], dtype=x.dtype)) * -1e10

    # compute self attention output
    x = attention(q, k, v, mask)
    
    # output projection
    x = linear(x, **c_proj)
    
    return x

# multi-head causal self attention function used in gpt-2 transformer blocks
def mh_c_s_a(x, c_attn, c_proj, n_head):
    
    # qkv projections
    x = linear(x, **c_attn)
    
    # split into q,k,v
    q,k,v = np.split(x, 3, axis = -1)
    
    # 3. Split each matrix into separate arrays for each "head"
    q_heads = np.split(q, n_head, axis=-1)  # List of small Q matrices
    k_heads = np.split(k, n_head, axis=-1)  # List of small K matrices
    v_heads = np.split(v, n_head, axis=-1)  # List of small V matrices

    # mask values for causal attention (upper triangular mask)
    mask = (1- np.tri(x.shape[0], dtype=x.dtype)) * -1e10

    # perform attention over each head
    output_heads = [attention(qh, kh, vh, mask) for qh, kh, vh in zip(q_heads, k_heads, v_heads)]

    #merge output of heads back together
    x = np.hstack(output_heads)
    
    # output projection
    x = linear(x, **c_proj)
    
    return x

## simple generation function that takes in a list of input token ids and generates new tokens auto-regressively using the gpt-2 model
def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)
