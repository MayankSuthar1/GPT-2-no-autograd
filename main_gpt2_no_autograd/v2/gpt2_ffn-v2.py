"""V-2 of gpt2.py with feed forward network implemented as a separate class.
Used in the transformer block instead of being implemented as a function.
and didnt used pre-trained weights of gpt-2
"""## converting into class make the easier to implement the backward pass for the backpropagation 
import numpy as np
import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-2")

# Helper functions
def linear(input, weight, bias=None):
    # Weight matrix should be in the shape of (output_features, input_features)
    # Perform matrix multiplication accordingly
    output = np.dot(input, weight.T)  # No additional transpose needed on the weight matrix
    if bias is not None:
        output += bias
    return output

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def embedding(vocab_size, embedding_dim):
    return np.random.randn(vocab_size, embedding_dim)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def layernorm(x, gamma, beta, epsilon=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    normalized_x = (x - mean) / np.sqrt(variance + epsilon)
    return gamma * normalized_x + beta


# Parameters setup
n_embd = 768
n_layer = 12
n_head = 12
n_positions = 1024
vocab_size = 50257

class AttentionHead:
    def __init__(self, embd_size, head_size):
        self.W_q = np.random.randn(head_size, embd_size)
        self.W_k = np.random.randn(head_size, embd_size)
        self.W_v = np.random.randn(head_size, embd_size)
        self.tril = np.tril(np.ones((n_positions, n_positions)))

    def forward(self, x):
        q = linear(x, self.W_q)
        k = linear(x, self.W_k)
        v = linear(x, self.W_v)

        k_transpose = k.transpose(0, 2, 1)  # Transposing for proper matrix multiplication
        wei = np.matmul(q, k_transpose) / np.sqrt(k.shape[-1])
        wei = np.where(self.tril[:x.shape[1], :x.shape[1]] == 0, -np.inf, wei)
        wei = softmax(wei)
        out = np.matmul(wei, v)
        return out

class MultiHeadAttention:
    def __init__(self, embd_size, num_heads):
        self.heads = [AttentionHead(embd_size, embd_size // num_heads) for _ in range(num_heads)]
        self.proj = np.random.randn(embd_size, embd_size)  # Correct projection matrix

    def forward(self, x):
        head_outputs = [head.forward(x) for head in self.heads]
        concat = np.concatenate(head_outputs, axis=2)
        return linear(concat, self.proj)

class FeedForward:
    def __init__(self, n_embd):
        # Adjust dimension order for weight matrices
        self.W1 = np.random.randn(n_embd * 4, n_embd)  # [output_features, input_features]
        self.b1 = np.zeros(n_embd * 4)
        self.W2 = np.random.randn(n_embd, n_embd * 4)  # [output_features, input_features]
        self.b2 = np.zeros(n_embd)

    def forward(self, x):
        x = linear(x, self.W1, self.b1)
        x = gelu(x)
        x = linear(x, self.W2, self.b2)
        return x

class TransformerBlock:
    def __init__(self, n_embd, n_head):
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ffn = FeedForward(n_embd)
        self.ln1 = np.ones(n_embd), np.zeros(n_embd)  # Gamma, Beta for LayerNorm
        self.ln2 = np.ones(n_embd), np.zeros(n_embd)

    def forward(self, x):
        x = x + self.attn.forward(layernorm(x, *self.ln1))
        x = x + self.ffn.forward(layernorm(x, *self.ln2))
        return x

class GPT2:
    def __init__(self):
        self.token_emb = embedding(vocab_size, n_embd)
        self.pos_emb = embedding(n_positions, n_embd)
        self.transformer_blocks = [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        
    def forward(self, inputs):
        x = self.token_emb[inputs] + self.pos_emb[np.arange(inputs.shape[1]) % n_positions]
        for block in self.transformer_blocks:
            x = block.forward(x)
        x = x @ self.token_emb.T  # Projection to vocabulary size
        return x


# Create a GPT2 model instance
model = GPT2()

# Processing a sentence
sentence = "Hello There! How are you doing today?"
token_ids = tokenizer.encode(sentence)  # Tokenizing the sentence

# Ensure token_ids are within our assumed vocabulary size
token_ids = [tid % vocab_size for tid in token_ids]

# Running through the model
inputs = np.array([token_ids])  # Shape (1, sequence_length)
output = model.forward(inputs)

print(output.shape)  # Output shape should be (1, sequence_length, n_embd)
print(output)  # Print the output logits for each token in the vocabulary
text_output = tokenizer.decode(np.argmax(output, axis=-1).flatten())
print(text_output)  # Print the decoded output text (may not be meaningful due to random