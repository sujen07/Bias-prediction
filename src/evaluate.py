from train import compute_classifier_accuracy, device
from tokenizer import encoding
import torch
from models.transformers import Classifier, EncoderModel, BiasTokenDataset
from preprocess import preprocess_csv
from torch.utils.data import DataLoader

embed_size = 50
block_size = 128
num_layers = 4
num_heads = 2
hidden_size=128


def sammple_incorrect_predictions(incorrect_predictions):
    for i, example in enumerate(incorrect_predictions[:5]):
        bias_token_indices = example['biased_tokens']
        bias_words = []
        for ind in bias_token_indices:
            token = example['input_ids'][ind]
            bias_token = encoding.decode([token])
            bias_words.append(bias_token)

        print(f"Example {i + 1}:")
        print(f"  Text: {encoding.decode(example['input_ids'])}")
        print(f"Predicted Bias Tokens: ' {bias_words}")
        print(f"  True Label: {example['true_label']}")
        print(f"  Predicted Label: {example['predicted_label']}")
        

if __name__ == '__main__':
    encoder = EncoderModel(encoding.n_vocab, embed_size, block_size, num_heads, num_layers)
    model = Classifier(encoder, embed_size, hidden_size, n_output=2)
    model.load_state_dict(torch.load('./src/models/final_model.pth'))
    model.eval()
    model = model.to(device)
    
    train_df, test_df, val_df = preprocess_csv()
    
    test_dataset = BiasTokenDataset(test_df, max_length=block_size)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    
    accuracy, incorrect_predictions = compute_classifier_accuracy(model, test_loader)
    sammple_incorrect_predictions(incorrect_predictions)