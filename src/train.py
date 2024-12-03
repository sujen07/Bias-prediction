from torch.utils.data import DataLoader
from models.baseline import MLP, TextBiasDataset
from models.transformers import EncoderModel, Classifier, BiasTokenDataset
from preprocess import preprocess_csv
from tokenizer import encoding
import torch

device = 'cpu'
embed_size = 50
block_size = 32
num_layers = 4
num_heads = 2


def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            bias_tokens = batch['bias_tokens'].to(device)
            labels = batch['label'].to(device)
            outputs, bias_predictions = classifier(input_ids, bias_tokens)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy
    
def compute_baseline_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy
    
    
def compute_loss(outputs, labels, bias_predictions=None, bias_labels=None):
    primary_loss = criterion(outputs, labels)
    if bias_predictions is not None and bias_labels is not None:
        auxiliary_loss = bias_criterion(bias_predictions, bias_labels.float())
        return primary_loss + auxiliary_loss * 0.5 
    return primary_loss


def train_baseline(train_loader, val_loader, model, optimizer):
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

            train_accuracy = compute_baseline_accuracy(model, train_loader)
            test_accuracy = compute_baseline_accuracy(model, val_loader)
            print(f"training accuracy: {train_accuracy}")
            print(f"Val accuracy: {test_accuracy}")



def train_model(model, train_loader, val_loader, optimizer, num_epochs):
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

            train_accuracy = compute_classifier_accuracy(model, train_loader)
            test_accuracy = compute_classifier_accuracy(model, val_loader)
            print(f"training accuracy: {train_accuracy}")
            print(f"Val accuracy: {test_accuracy}")


if __name__ == "__main__":
    train_df, test_df, val_df = preprocess_csv()

    train_dataset = BiasTokenDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = BiasTokenDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)


    model = MLP(encoding.n_vocab, 256, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    bias_criterion = torch.nn.BCELoss()
    num_epochs = 10
    train_baseline(train_loader, val_loader, model, optimizer)

    train_dataset = BiasTokenDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = BiasTokenDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    encoder = EncoderModel(encoding.n_vocab, embed_size, block_size, num_heads, num_layers)
    model = Classifier(encoder, 50, 128, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 20