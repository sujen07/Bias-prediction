from torch.utils.data import DataLoader
from models.baseline import MLP, TextBiasDataset
from models.transformers import EncoderModel, Classifier, BiasTokenDataset
from preprocess import preprocess_csv
from tokenizer import encoding
import torch
import matplotlib.pyplot as plt
import time

device = 'cpu'
embed_size = 50
block_size = 128
num_layers = 4
num_heads = 2
hidden_size=128


def compute_classifier_accuracy(classifier, data_loader, threshold=0.5):
    """ Compute accuracy and collect examples predicted incorrectly, 
        including the biased words based on the threshold.
    """
    classifier.eval()
    total_correct = 0
    total_samples = 0
    incorrect_predictions = []  # Store incorrect examples
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            bias_tokens = batch['bias_tokens'].to(device)
            labels = batch['label'].to(device)

            # Get model output and bias predictions
            outputs, bias_predictions = classifier(input_ids, bias_tokens)
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Collect incorrect predictions
            incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0]
            for idx in incorrect_indices:
                # Find biased tokens (where bias_predictions > threshold)
                biased_tokens = (bias_predictions[idx] > threshold).nonzero(as_tuple=True)[0]
                
                # Add the incorrect example and biased tokens to the dictionary
                incorrect_predictions.append({
                    "input_ids": input_ids[idx].cpu().tolist(),
                    "bias_tokens": bias_tokens[idx].cpu().tolist(),
                    "true_label": labels[idx].item(),
                    "predicted_label": predicted[idx].item(),
                    "biased_tokens": biased_tokens.cpu().tolist()  # Indices of biased tokens
                })

        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy, incorrect_predictions


    
def compute_baseline_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    incorrect_predictions = []
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)

            incorrect_indices = (predicted != Y).nonzero(as_tuple=True)[0]
            for idx in incorrect_indices:
                
                # Add the incorrect example and biased tokens to the dictionary
                incorrect_predictions.append({
                    "input_ids": X[idx].cpu().tolist(),
                    "true_label": Y[idx].item(),
                    "predicted_label": predicted[idx].item(),
                })

        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy, incorrect_predictions
    
    
def compute_loss(outputs, labels, bias_predictions=None, bias_labels=None):
    primary_loss = criterion(outputs, labels)
    if bias_predictions is not None and bias_labels is not None:
        auxiliary_loss = bias_criterion(bias_predictions, bias_labels.float())
        return primary_loss + auxiliary_loss * 0.5 
    return primary_loss


def train_baseline(train_loader, val_loader, model, optimizer, num_epochs):
    start_time = time.time()
    for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)  
                loss = criterion(outputs, yb)
                optimizer.zero_grad()  
                loss.backward()  
                optimizer.step() 
                
                total_loss += loss.item() 
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

            train_accuracy, incorrect_predictions = compute_baseline_accuracy(model, train_loader)
            test_accuracy, incorrect_predictions = compute_baseline_accuracy(model, val_loader)
            print(f"training accuracy: {train_accuracy}")
            print(f"Val accuracy: {test_accuracy}")
    end_time = time.time()
    print(f'Total Time taken for basline: {end_time - start_time}')



def train_model(model, train_loader, val_loader, optimizer, num_epochs):
    train_accuracies = []
    val_accuracies = []
    
    start_time = time.time()

    for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                bias_tokens = batch['bias_tokens'].to(device)
                labels = batch['label'].to(device)
                outputs, bias_predictions = model(input_ids, bias_tokens)
                loss = compute_loss(outputs, labels, bias_predictions, bias_tokens)
                optimizer.zero_grad()  
                loss.backward()  
                optimizer.step() 
                
                total_loss += loss.item() 
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

            train_accuracy, incorrect_predictions = compute_classifier_accuracy(model, train_loader)
            val_accuracy, incorrect_predictions = compute_classifier_accuracy(model, val_loader)

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)


            print(f"training accuracy: {train_accuracy}")
            print(f"Val accuracy: {val_accuracy}")
    end_time = time.time()
    print(f'Total Time taken for training: {end_time - start_time}')
    return incorrect_predictions



if __name__ == "__main__":
    train_df, test_df, val_df = preprocess_csv()

    train_dataset = TextBiasDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = TextBiasDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)


    baseline_model = MLP(encoding.n_vocab, embed_size, hidden_size)
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    bias_criterion = torch.nn.BCELoss()
    num_epochs = 10
    train_baseline(train_loader, val_loader, baseline_model, optimizer, num_epochs)

    train_dataset = BiasTokenDataset(train_df, max_length=block_size)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = BiasTokenDataset(val_df, max_length=block_size)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)


    encoder = EncoderModel(encoding.n_vocab, embed_size, block_size, num_heads, num_layers).to(device)
    model = Classifier(encoder, embed_size, hidden_size, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 20
    incorrect_predictions = train_model(model, train_loader, val_loader, optimizer, num_epochs)
    test_accuracy, incorrect_predictions = compute_classifier_accuracy(model, val_loader)
    
    torch.save(model.state_dict(), './src/models/final_model.pth')