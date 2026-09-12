import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.W_1h = nn.Linear(embedding_dim, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)

        self.hidden_size = hidden_size

    def forward(self, x, h_prev):
        x = self.embedding(x)

        h_t = torch.tanh(self.W_1h(x) + self.W_hh(h_prev))

        y_t = self.W_hy(h_t)

        return y_t, h_t

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


words = ["I", "love", "machine", "learning"]

word_to_id = {"I": 0, "love": 1, "machine": 2, "learning": 3}

id_to_word = {v: k for k, v in word_to_id.items()}


inputs = torch.tensor([0, 1, 2])
targets = torch.tensor([1, 2, 3])


model = SimpleRNN(vocab_size=4, embedding_dim=8, hidden_size=20, output_size=4)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# =========== Training

for epoch in range(1000):
    h = model.init_hidden(batch_size=1)

    total_loss = 0

    for x, target in zip(inputs, targets):
        x = x.unsqueeze(0)

        y, h = model(x, h)

        loss = loss_fn(y, target.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        h = h.detach()

        total_loss += loss.item()


# ==== Prediction

word = "I"

x = torch.tensor([word_to_id[word]])

h = model.init_hidden(batch_size=1)

y, h = model(x, h)

predicted_id = torch.argmax(y, dim=1).item()

predicted_word = id_to_word[predicted_id]

print("Input:", word)
print("Predicted next word:", predicted_word)
