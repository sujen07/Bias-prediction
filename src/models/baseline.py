import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizer import Tokenizer




class TextBiasDataset(Dataset):
    def __init__(self, dataframe):

        self.dataframe = dataframe
        self.tokenizer = Tokenizer()
        self.label_mapping = {'Non-biased': 0, 'Biased': 1}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract text and label
        text = self.dataframe.iloc[idx]['text']
        label_str = self.dataframe.iloc[idx]['label_bias']
        
        # Tokenize text
        tokenized = self.tokenizer.tokenize(
            text,
            padding=True, 
            max_length=32,  
        )
        
        label = torch.tensor(self.label_mapping[label_str], dtype=torch.long)
        
        return tokenized, label


class MLP(torch.nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

