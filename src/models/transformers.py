# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from tokenizer import Tokenizer



class BiasTokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer=None, max_length=32):
        self.dataframe = dataframe
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.label_mapping = {'Non-biased': 0, 'Biased': 1}
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract text, bias words, and label
        text = self.dataframe.iloc[idx]['text']
        label_str = self.dataframe.iloc[idx]['label_bias']
        bias_words = eval(self.dataframe.iloc[idx]['biased_words'])  

        # Tokenize the text using the tiktoken tokenizer
        tokenized = self.tokenizer.tokenize(text, max_length=self.max_length, padding=True)

        # Initialize bias tokens to 0s
        bias_tokens = torch.zeros(self.max_length, dtype=torch.long)

        # Decode each token into a subword and compare to the bias words
        decoded_tokens = [self.tokenizer.decode([token]) for token in tokenized]

        for i, subword in enumerate(decoded_tokens):
            # Check if the subword matches any bias word (case insensitive)
            if any(bias_word.lower() in subword.lower() for bias_word in bias_words):
                bias_tokens[i] = 1

        # Convert label to tensor
        label = torch.tensor(self.label_mapping[label_str], dtype=torch.long)

        # Return tokenized input, bias tokens, and label
        return {
            'input_ids': tokenized,
            'bias_tokens': bias_tokens,
            'label': label
        }



class Head(nn.Module):
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.values = nn.Linear(embed_size, head_size, bias=False)
        self.keys = nn.Linear(embed_size, head_size, bias=False)
        self.head_size = head_size

    def forward(self, x):
        Q = self.query(x)
        V = self.values(x)
        K = self.keys(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_size)


        wei = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(wei, V)
        return out, wei
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, head_size, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(embed_size=embed_size, head_size=head_size) 
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(head_size * n_heads, embed_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = []
        attn_maps = []
        for h in self.heads:
            head_out, wei = h(x)
            out.append(head_out)
            attn_maps.append(wei)
        out = torch.cat(out, dim=-1)
        out = self.proj(out)
        attn_maps = torch.stack(attn_maps, dim=0)
        return out, attn_maps

class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, embed_size*4)
        self.fc2 = nn.Linear(embed_size*4, embed_size)
        self.dropout = nn.Dropout(0.4)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, embed_size, n_heads):
        super().__init__()
        head_size = embed_size // n_heads
        self.sa = MultiHeadAttention(embed_size, head_size, n_heads)
        self.ffw = FeedForward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        ln1_x = self.ln1(x)
        sa_out, attn_maps = self.sa(ln1_x)
        x = x + sa_out
        x = x + self.ffw(self.ln2(x))
        return x, attn_maps
    


class EncoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size, n_heads, n_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.bias_embedding = nn.Embedding(2, embed_size)  # Binary (0: non-bias, 1: bias)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.blocks = nn.ModuleList([Block(embed_size, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, bias_tokens):
        B, T = input_ids.shape
        token_emb = self.token_embedding(input_ids)  # Token embeddings
        bias_emb = self.bias_embedding(bias_tokens)  # Bias token embeddings
        pos_emb = self.position_embedding(torch.arange(T, device=input_ids.device))  # Positional embeddings

        # Fuse embeddings
        x = self.ln_f(token_emb + bias_emb + pos_emb)
        x = self.dropout(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.ln_f(x)
        return x

    



class Classifier(nn.Module):
    def __init__(self, encoder, embed_size, n_hidden, n_output):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(embed_size, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.token_bias_classifier = nn.Linear(embed_size, 1)  # Per-token bias classification
        self.dropout = nn.Dropout(0.4)

    def forward(self, input_ids, bias_tokens):
        # Encoder output
        encoder_output = self.encoder(input_ids, bias_tokens)  # Shape: (batch_size, seq_length, embed_size)
        
        # Aggregate sequence representation for global classification
        seq_rep = encoder_output.mean(dim=1)  # Shape: (batch_size, embed_size)
        x = F.relu(self.fc1(seq_rep))
        x = self.dropout(x)
        classification_output = self.fc2(x)  # Shape: (batch_size, n_output)

        # Per-token bias prediction
        bias_prediction = torch.sigmoid(self.token_bias_classifier(encoder_output)).squeeze(-1)  # Shape: (batch_size, seq_length)

        return classification_output, bias_prediction


    

