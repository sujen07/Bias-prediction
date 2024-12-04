import tiktoken
import torch

encoding = tiktoken.get_encoding("cl100k_base")



class Tokenizer():
    def __init__(self):
        self.tokenizer = encoding

    def tokenize(self, text, max_length=None, padding=False):
        token_ids = self.tokenizer.encode(text)

        # Handle truncation
        if max_length is not None:
            token_ids = token_ids[:max_length]

        # Handle padding
        if padding and max_length is not None:
            # Pad the sequence to max_length
            pad_token_id = self.tokenizer.eot_token if hasattr(self.tokenizer, 'eot_token') else 0
            token_ids = token_ids + [pad_token_id] * (max_length - len(token_ids))

        # Convert to PyTorch tensor
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, tokens):
        return encoding.decode(tokens)