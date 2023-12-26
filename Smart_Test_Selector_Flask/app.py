from flask import Flask, render_template, request, jsonify
import torch
from torch.nn.functional import softmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

app = Flask(__name__)

df1 = pd.read_csv('csv.csv')

code_changes = df1['Code Changes'].tolist()
smart_test_selection = df1['Smart Test Selection Mechanism'].tolist()

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(code_changes)
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences
input_sequences = tokenizer.texts_to_sequences(code_changes)

# Padding sequences for consistent input size
input_sequences = pad_sequences(input_sequences)

# Convert 'smart_test_selection' to labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(smart_test_selection)

# Define PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Create PyTorch Dataset and DataLoader
dataset = CustomDataset(input_sequences, encoded_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class PyModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, output_size):
        super(PyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        _, (x, _) = self.lstm(x)
        x = x.squeeze(0)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
# Load the pre-trained PyTorch model and label encoder
loaded_pymodel = PyModel(vocab_size=total_words, embedding_dim=100, lstm_hidden_dim=100, output_size=len(set(encoded_labels)))
loaded_pymodel.load_state_dict(torch.load('pymodel.pth'))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(code_changes)

label_encoder = LabelEncoder()
label_encoder.fit_transform(smart_test_selection)

def predict_label(new_code_snippet):
    new_code_sequence = tokenizer.texts_to_sequences([new_code_snippet])
    padded_new_code_sequence = pad_sequences(new_code_sequence, maxlen=input_sequences.shape[1])
    input_tensor = torch.LongTensor(padded_new_code_sequence)

    with torch.no_grad():
        outputs = loaded_pymodel(input_tensor)
        predicted_class_index = torch.argmax(outputs)

    predicted_label = label_encoder.inverse_transform([predicted_class_index.item()])[0]
    return predicted_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        new_code_snippet = request.form['new_code_snippet']
        predicted_label = predict_label(new_code_snippet)
        return render_template('index.html', prediction=predicted_label)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)