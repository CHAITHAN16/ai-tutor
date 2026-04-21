import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.stem.porter import PorterStemmer
from pathlib import Path

nltk.download("punkt", quiet=True)

stemmer = PorterStemmer()


def tokenize(sentence: str):
    return nltk.word_tokenize(sentence)


def stem(word: str):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


def main():
    base_path = Path(__file__).parent
    with (base_path / "intents.json").open("r", encoding="utf-8") as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            tokenized = tokenize(pattern)
            all_words.extend(tokenized)
            xy.append((tokenized, tag))

    ignore_words = ["?", "!", ".", ","]
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        y_train.append(tags.index(tag))

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.longlong)

    input_size = len(all_words)
    hidden_size = 8
    output_size = len(tags)
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 1000

    model = NeuralNet(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        inputs = torch.from_numpy(X_train)
        labels = torch.from_numpy(y_train)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    model_data = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags,
        "model_state": model.state_dict(),
    }

    torch.save(model_data, base_path / "model.pth")
    print("Training complete. Saved model.pth")


if __name__ == "__main__":
    main()
