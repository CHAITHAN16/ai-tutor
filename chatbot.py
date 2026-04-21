import json
import random
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from groq import Groq
from nltk.stem.porter import PorterStemmer

# Initialize Groq Client
client = Groq(
    api_key="Your api key here")
stemmer = PorterStemmer()


def tokenize(sentence):
    return re.findall(r"\b\w+\b", sentence.lower())


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


base_path = Path(__file__).parent
with open(base_path / "intents.json", "r") as f:
    intents = json.load(f)

data = torch.load(base_path / "model.pth")
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Global chat history for memory
chat_history = [
    {"role": "system", "content": "You are a concise CSE Tutor. Explain concepts in maximum 3-4 sentences. Use bullet points for steps and wrap all code snippets in backticks. Do not include long introductions or concluding remarks."}
]


def ask_llm(user_input):
    global chat_history
    try:
        # Add user message to history
        chat_history.append({"role": "user", "content": user_input})

        # Keep history manageable (last 10 messages)
        if len(chat_history) > 11:
            chat_history = [chat_history[0]] + chat_history[-10:]

        chat = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=chat_history,
            max_tokens=150,
            temperature=0.5
        )
        content = chat.choices[0].message.content

        if isinstance(content, str) and content.strip():
            # Add AI response to history
            chat_history.append(
                {"role": "assistant", "content": content.strip()})
            return content.strip()
        return "No response from AI"
    except Exception as e:
        print("ERROR:", e)
        return f"ERROR: {str(e)}"


def predict(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    confidence = probs[0][predicted.item()].item()

    local_response = None
    for intent in intents["intents"]:
        if tag == intent["tag"] and intent["responses"]:
            local_response = random.choice(intent["responses"])
            break
    return {"tag": tag, "confidence": confidence, "response": local_response}


def get_response(msg):
    prediction = predict(msg)
    # If confidence is high, use local response (Fast/Free)
    if prediction["confidence"] > 0.75 and prediction["response"]:
        return prediction["response"]
    # Fallback to LLM with memory
    return ask_llm(msg)
