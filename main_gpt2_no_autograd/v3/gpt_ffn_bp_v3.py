"""V-2 of gpt2.py with feed forward network implemented as a separate class.
Used in the transformer block instead of being implemented as a function.
and didnt used pre-trained weights of gpt-2
"""## converting into class make the easier to implement the backward pass for the backpropagation 
import numpy as np


# Helper functions
class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.weight = np.random.randn(vocab_size, embed_dim) * 0.02
        
    def forward(self, inputs):
        self.inputs = inputs
        return self.weight[inputs]
        
    def backward(self, grad_output):
        self.grad_weight = np.zeros_like(self.weight)
        np.add.at(self.grad_weight, self.inputs, grad_output)
        # We do not pass gradients back past embeddings since the inputs are integers
        return None
    
class Linear:
    def __init__(self, input_dim, output_dim):
        self.weight = np.random.randn(output_dim, input_dim) * 0.02 # [output_features, input_features]
        self.bias = np.zeros(output_dim)

    def forward(self, x):
        self.x = x  # Store input for backward pass
        return np.dot(x, self.weight.T) + self.bias # Perform matrix multiplication

    def backward(self, grad_output):
        # grad_output is the gradient of the loss with respect to this layer's output
        
        # 1. Gradients with respect to weights and biases
        # Flatten the batch and seq_len dimensions to 2D matrices
        x_reshaped = self.x.reshape(-1, self.x.shape[-1])
        grad_out_reshaped = grad_output.reshape(-1, grad_output.shape[-1])
        
        # Weight gradient should match self.weight shape: (output_dim, input_dim)
        grad_w = np.dot(grad_out_reshaped.T, x_reshaped) 
        grad_b = np.sum(grad_out_reshaped, axis=0)
        self.grad_weight = grad_w
        self.grad_bias = grad_b
        
        # 2. Gradient of the input (to pass back to the previous layer)
        # Note: grad_output is (..., output_dim) and self.weight is (output_dim, input_dim)
        # This correctly projects the gradient back to (..., input_dim)
        grad_input = np.dot(grad_output, self.weight)
        
        return grad_input, grad_w, grad_b                   

class GELU:
    def forward(self, x):
        self.x = x
        self.inner = np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))
        self.tanh = np.tanh(self.inner)
        self.cdf = 0.5 * (1.0 + self.tanh)
        return x * self.cdf

    def backward(self, grad_output):
        x = self.x
        # Approximation of derivative of GELU
        sech2 = 1.0 - self.tanh ** 2
        pdf = 0.5 * np.sqrt(2 / np.pi) * sech2 * (1.0 + 3.0 * 0.044715 * np.power(x, 2))
        grad_x = self.cdf + x * pdf
        return grad_output * grad_x

class Softmax:
    def forward(self, x):
        # Shift the input for numerical stability (doesn't change softmax output)
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x_shifted)
        # Cache the output probabilities for the backward pass
        self.output = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        # grad_output is dL/dy, self.output is y
        # Compute the dot product of the gradient and the output along the last axis
        sum_grad_y = np.sum(grad_output * self.output, axis=-1, keepdims=True)
        
        # Apply the simplified Jacobian-vector product formula
        grad_input = self.output * (grad_output - sum_grad_y)
        return grad_input

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.x_norm = (x - self.mean) / self.std
        return self.gamma * self.x_norm + self.beta

    def backward(self, grad_output):
        N = self.x.shape[-1]
        
        # Gradients for gamma and beta (summing across batch and seq_len dims)
        axes = tuple(range(grad_output.ndim - 1))
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=axes)
        self.grad_beta = np.sum(grad_output, axis=axes)
        
        # Gradient for input passed down the network
        dx_norm = grad_output * self.gamma
        dx = (1.0 / N) / self.std * (
            N * dx_norm -
            np.sum(dx_norm, axis=-1, keepdims=True) -
            self.x_norm * np.sum(dx_norm * self.x_norm, axis=-1, keepdims=True)
        )
        return dx


    
# Parameters setup
n_embd = 768
n_layer = 12
n_head = 12
n_positions = 1024
vocab_size = 50257

class AttentionHead:
    def __init__(self, embd_size, head_size, n_positions=1024):
        # We use the Linear class instead of standalone weight matrices 
        # so they handle their own gradients during the backward pass!
        self.q_proj = Linear(embd_size, head_size)
        self.k_proj = Linear(embd_size, head_size)
        self.v_proj = Linear(embd_size, head_size)
        self.softmax_layer = Softmax()
        self.tril = np.tril(np.ones((n_positions, n_positions)))
        self.scale = np.sqrt(head_size)

    def forward(self, x):
        self.x = x
        self.seq_len = x.shape[1]
        
        # 1. Linear projections
        self.q = self.q_proj.forward(x)
        self.k = self.k_proj.forward(x)
        self.v = self.v_proj.forward(x)

        # 2. Scaled Dot-Product Attention
        self.k_T = self.k.transpose(0, 2, 1)  # (batch, head_size, seq_len)
        self.attn_scores = np.matmul(self.q, self.k_T) / self.scale
        
        # 3. Masking
        mask = self.tril[:self.seq_len, :self.seq_len]
        self.attn_scores = np.where(mask == 0, -np.inf, self.attn_scores)
        
        # 4. Softmax
        self.wei = self.softmax_layer.forward(self.attn_scores)
        
        # 5. Multiply weights by Values
        out = np.matmul(self.wei, self.v)
        return out
    
    def backward(self, grad_output):
        """
        grad_output shape: (batch_size, seq_len, head_size)
        """
        # 1. Backprop through Context Vector = wei @ v
        # d_v = wei^T @ grad_output
        grad_v = np.matmul(self.wei.transpose(0, 2, 1), grad_output)
        # d_wei = grad_output @ v^T
        grad_wei = np.matmul(grad_output, self.v.transpose(0, 2, 1))

        # 2. Backprop through Softmax
        grad_attn_scores = self.softmax_layer.backward(grad_wei)

        # 3. Backprop through Masking 
        # The gradient is 0 where the mask was applied (where tril == 0)
        mask = self.tril[:self.seq_len, :self.seq_len]
        grad_attn_scores = np.where(mask == 0, 0, grad_attn_scores)

        # 4. Backprop through Scale
        grad_attn_scores = grad_attn_scores / self.scale

        # 5. Backprop through dot product: attn_scores = q @ k^T
        # d_q = grad_attn_scores @ k
        grad_q = np.matmul(grad_attn_scores, self.k)
        # d_k = grad_attn_scores^T @ q
        grad_k = np.matmul(grad_attn_scores.transpose(0, 2, 1), self.q)

        # 6. Backprop through linear layers
        grad_x_v, _, _ = self.v_proj.backward(grad_v)
        grad_x_k, _, _ = self.k_proj.backward(grad_k)
        grad_x_q, _, _ = self.q_proj.backward(grad_q)

        # 7. Provide final gradient to the input by accumulating the branches
        grad_input = grad_x_q + grad_x_k + grad_x_v
        
        return grad_input

class MultiHeadAttention:
    def __init__(self, embd_size, num_heads, n_positions=1024):
        assert embd_size % num_heads == 0, "embd_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_size = embd_size // num_heads
        
        # Instantiate the individual attention heads
        self.heads = [AttentionHead(embd_size, self.head_size, n_positions) for _ in range(num_heads)]
        
        # Replace the numpy matrix with the stateful Linear class
        self.proj = Linear(embd_size, embd_size)

    def forward(self, x):
        # Pass input through all heads and collect outputs
        head_outputs = [head.forward(x) for head in self.heads]
        
        # Concatenate head outputs along the embedding dimension (axis=2)
        # Result shape: (batch_size, seq_len, embd_size)
        concat = np.concatenate(head_outputs, axis=2)
        
        # Pass the concatenated states through the final linear projection
        result = self.proj.forward(concat)
        return result

    def backward(self, grad_output):
        # 1. Backprop through the final linear projection
        grad_concat, _, _ = self.proj.backward(grad_output)
        
        # 2. Split the gradient back into chunks for each head
        # Because we concatenated along axis=2, we split along axis=2
        # `grad_concat` has shape (batch_size, seq_len, embd_size)
        # `grad_heads` will be a list of num_heads arrays, each of shape (batch, seq, head_size)
        grad_heads = np.split(grad_concat, self.num_heads, axis=2)
        
        # 3. Backprop through each individual head
        # Each head receives its respective slice of the gradient
        head_input_grads = []
        for i, head in enumerate(self.heads):
            # The backward pass of the head computes the gradient w.r.t the input x
            grad_x = head.backward(grad_heads[i])
            head_input_grads.append(grad_x)
            
        # 4. Accumulate (sum) the gradients from all heads
        # Because all heads received the identical input 'x' in the forward pass,
        # their gradients w.r.t 'x' simply add together (multivariate chain rule).
        grad_input = sum(head_input_grads)
        
        return grad_input

class FeedForward:
    def __init__(self, n_embd):
        self.fc1 = Linear(n_embd, n_embd * 4)
        self.gelu = GELU()
        self.fc2 = Linear(n_embd * 4, n_embd)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.gelu.forward(x)
        x = self.fc2.forward(x)
        return x
        
    def backward(self, grad_output):
        grad = self.fc2.backward(grad_output)[0]  # Take only input grad
        grad = self.gelu.backward(grad)
        grad = self.fc1.backward(grad)[0]
        return grad

class TransformerBlock:
    def __init__(self, n_embd, n_head, n_positions=1024):
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, n_positions)
        self.ln2 = LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)

    def forward(self, x):
        self.skip1 = x
        self.ln1_out = self.ln1.forward(x)
        self.attn_out = self.attn.forward(self.ln1_out)
        self.x_mid = self.skip1 + self.attn_out
        
        self.skip2 = self.x_mid
        self.ln2_out = self.ln2.forward(self.x_mid)
        self.ffn_out = self.ffn.forward(self.ln2_out)
        x_out = self.skip2 + self.ffn_out
        return x_out
        
    def backward(self, grad_output):
        # ffn residual branch
        grad_ffn = self.ffn.backward(grad_output)
        grad_ln2 = self.ln2.backward(grad_ffn)
        grad_mid = grad_output + grad_ln2
        
        # attn residual branch
        grad_attn = self.attn.backward(grad_mid)
        grad_ln1 = self.ln1.backward(grad_attn)
        grad_in = grad_mid + grad_ln1
        
        return grad_in

class GPT2:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, n_positions):
        print(f"[DEBUG] Initializing GPT2 model: vocab_size={vocab_size}, n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}")
        self.token_emb = Embedding(vocab_size, n_embd)
        self.pos_emb = Embedding(n_positions, n_embd)
        self.blocks = [TransformerBlock(n_embd, n_head, n_positions) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = Linear(n_embd, vocab_size) # Alternatively: tie weights here
        
    def forward(self, inputs):
        self.seq_len = inputs.shape[1]
        positions = np.arange(self.seq_len)
        
        x = self.token_emb.forward(inputs) + self.pos_emb.forward(positions)
        
        for block in self.blocks:
            x = block.forward(x)
            
        x = self.ln_f.forward(x)
        logits = self.lm_head.forward(x)
        return logits
        
    def backward(self, grad_logits):
        # print(f"[DEBUG] GPT2 backward pass called with grad_logits shape: {grad_logits.shape}")
        grad_x = self.lm_head.backward(grad_logits)[0]
        grad_x = self.ln_f.backward(grad_x)
        
        for block in reversed(self.blocks):
            grad_x = block.backward(grad_x)
            
        # Distribute gradient back to embedding tables (position and token map equivalently on index)
        self.token_emb.backward(grad_x)
        
        # Mean across batch dim for position embedding
        grad_pos = np.sum(grad_x, axis=0)
        self.pos_emb.backward(grad_pos)

class CrossEntropyLoss:
    def __init__(self):
        self.softmax = Softmax()
        
    def forward(self, logits, targets):
        """
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len) integer indices
        """
        self.original_shape = logits.shape
        
        # Flatten batch and sequence dimensions
        # logits_flat becomes (batch * seq_len, vocab_size)
        # targets_flat becomes (batch * seq_len,)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        
        # Numerical stability: subtract max
        shifted_logits = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
        
        self.probs = self.softmax.forward(shifted_logits)
        self.targets = targets_flat
        
        # Calculate negative log likelihood
        N = logits_flat.shape[0] # Total number of tokens (batch * seq_len)
        correct_probs = self.probs[np.arange(N), self.targets]
        loss = -np.sum(np.log(correct_probs + 1e-10)) / N
        return loss
        
    def backward(self):
        # print("[DEBUG] CrossEntropyLoss backward pass called.")
        N = self.probs.shape[0] # Total number of tokens
        grad_logits = self.probs.copy()
        
        # The gradient of CrossEntropy + Softmax is (Probs - OneHotTargets)
        grad_logits[np.arange(N), self.targets] -= 1
        
        # Average the gradient over the number of tokens
        grad_logits = grad_logits / N
        
        # Reshape back to (batch, seq_len, vocab_size) 
        # to match what the last Linear layer expects
        return grad_logits.reshape(self.original_shape)

class AdamOptimizer:
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters # A list of dictionaries or objects holding w, b, grad_w, grad_b
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Initialize Adam momentum states
        self.m = {id(p): np.zeros_like(p.data) for p in parameters}
        self.v = {id(p): np.zeros_like(p.data) for p in parameters}
        
    def step(self):
        self.t += 1
        for p in self.parameters:
            if p.grad is None:
                continue
                
            # Update biased first moment estimate
            self.m[id(p)] = self.beta1 * self.m[id(p)] + (1 - self.beta1) * p.grad
            # Update biased second raw moment estimate
            self.v[id(p)] = self.beta2 * self.v[id(p)] + (1 - self.beta2) * (p.grad ** 2)
            
            # Compute bias-corrected first and second moment estimates
            m_hat = self.m[id(p)] / (1 - self.beta1 ** self.t)
            v_hat = self.v[id(p)] / (1 - self.beta2 ** self.t)
            
            # Update weights
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

if __name__ == "__main__":
    import tiktoken

    tokenizer = tiktoken.encoding_for_model("gpt-2")

    # Create a GPT2 model instance
    model = GPT2(vocab_size=50257, n_embd=768, n_head=12, n_layer=12, n_positions=1024)

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
