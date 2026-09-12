import numpy as np


text = """
the king is a man 
the queen is a woman
the king rules the kingdom
the queen rules the kingdom
the man is strong
the woman is strong
"""

tokens = text.lower().split()

vocab = sorted(set(tokens))

word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

vocab_size = len(vocab)

input_ids = [word_to_id[word] for word in tokens[:-1]]
target_ids = [word_to_id[word] for word in tokens[1:]]

hidden_size = 32

np.random.seed(42)

W_xh = np.random.randn(hidden_size, vocab_size) * 0.01
W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
W_hy = np.random.randn(vocab_size, hidden_size) * 0.01

b_h = np.zeros((hidden_size, 1))
b_y = np.zeros((vocab_size, 1))


def one_hot(word_id):
    x = np.zeros((vocab_size, 1))
    x[word_id] = 1
    return x


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


learning_rate = 0.05
epochs = 500

# Forward pass only version
for epoch in range(epochs):
    h = np.zeros((hidden_size, 1))

    total_loss = 0

    for input_id, target_id in zip(input_ids, target_ids):
        x = one_hot(input_id)
        h = np.tanh(W_xh @ x + W_hh @ h + b_h)

        logits = W_hy @ h + b_y

        probabilities = softmax(logits)

        loss = -np.log(probabilities[target_id, 0] + 1e-9)

        total_loss += loss

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


def predict_next_word(word):

    if word not in word_to_id:
        print("Unknown word:", word)
        return

    # Start with empty memory
    h = np.zeros((hidden_size, 1))

    # Convert word to one-hot
    x = one_hot(word_to_id[word])

    # RNN
    h = np.tanh(W_xh @ x + W_hh @ h + b_h)

    # Output
    logits = W_hy @ h + b_y

    # Probabilities
    probabilities = softmax(logits)

    # Highest probability
    predicted_id = np.argmax(probabilities)

    predicted_word = id_to_word[predicted_id]

    print("Input:", word)
    print("Predicted:", predicted_word)

    return predicted_word


predict_next_word("the")
