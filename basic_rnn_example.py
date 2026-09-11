import numpy as np

# Dim

input_size = 4
hidden_size = 3
vocab_size = 6

# parameter

np.random.seed(42)

W_xh = np.random.randn(hidden_size, input_size) * 0.1
W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
W_hy = np.random.randn(vocab_size, hidden_size) * 0.1

b_h = np.zeros((hidden_size, 1))
b_y = np.zeros((vocab_size, 1))

# Initial Hidden Size

h_prev = np.zeros((hidden_size, 1))

# 1st step
x = np.random.randn(input_size, 1)

h = np.tanh(W_xh @ x + W_hh @ h_prev + b_h)

# Z (output)
logits = W_hy @ h + b_y


# Softmax

exp_logits = np.exp(logits - np.max(logits))
probabilities = exp_logits / np.sum(exp_logits)

print("Hidden state:")
print(h)

print("\nProbabilities:")
print(probabilities)
