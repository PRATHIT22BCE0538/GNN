import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch_geometric.data import Data, Dataset


import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GCNConv 
from sklearn.cluster import KMeans
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a consistent size
    transforms.ToTensor(),            # Convert to tensor
])

class C_NMCDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for fold in ['fold_0', 'fold_1', 'fold_2']:
            fold_path = os.path.join(self.root_dir, 'C-NMC_training_data', fold)
            for label in ['all', 'hem']:
                label_path = os.path.join(fold_path, label)
                for filename in os.listdir(label_path):
                    if filename.endswith('.bmp') or filename.endswith('.jpg'):  # Check for image files
                        img_path = os.path.join(label_path, filename)
                        image = Image.open(img_path)
                        image = transform(image)  # Apply the transformations
                        self.data.append(image)
                        self.labels.append(0 if label == 'hem' else 1)  # Assign labels: 0 for 'hem', 1 for 'all'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Usage
dataset = C_NMCDataset('PKG - C-NMC 2019')
print(f'Total number of samples: {len(dataset)}')

# Get the first sample
image_tensor, label = dataset[0]
print(f'First sample shape: {image_tensor.shape}')  # Shape of the image tensor
print(f'First sample label: {label}')  

# Define your C_NMCDataset class here (as you already have it)

# Function to load and preprocess dataset
def load_dataset(root_dir):
    return C_NMCDataset(root_dir)

# Function to extract features using a pre-trained model
def extract_features(dataset):
    feature_extractor = models.resnet18(pretrained=True)
    feature_extractor.fc = nn.Identity()
    feature_extractor.eval()
    
    features = []
    labels = []

    with torch.no_grad():
        for img, label in dataset:
            img = img.unsqueeze(0)
            feature_vector = feature_extractor(img)
            features.append(feature_vector.numpy())
            labels.append(label)

    return np.vstack(features), np.array(labels)

# Function to construct a graph from features
from sklearn.cluster import KMeans

# Function to construct a graph from features using KMeans
def construct_graph(features, labels, num_clusters=100):
    num_nodes = len(features)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)

    edge_index = []

    # Construct edges within each cluster
    for cluster in range(num_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster)[0]
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                edge_index.append((cluster_indices[i], cluster_indices[j]))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=torch.tensor(features, dtype=torch.float32), edge_index=edge_index, y=torch.tensor(labels, dtype=torch.long))

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Training Function
def train(model, data, optimizer, criterion, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)  # Get model predictions
        loss = criterion(out, data.y)  # Pass both output and labels to the loss function
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        print("Data x shape:", data.x.shape)
        print("Data edge_index shape:", data.edge_index.shape)
        print("Data y shape:", data.y.shape)
        out = model(data)  # Ensure this is the correct way to call your model
        print("Output shape:", out.shape)  # Check output shape
        print("Data edge index max:", data.edge_index.max().item())  # Max edge index
        print("Data number of nodes:", data.num_nodes)  # Check the number of nodes
        pred = out.argmax(dim=1)  # Get the predicted classes
        acc = (pred == data.y).sum().item() / data.y.size(0)  # Calculate accuracy
        print(f'Accuracy: {acc * 100:.2f}%')
        print("---------------------------------------------------------------")
        print("---------------------------------------------------------------")
        return acc

def tune_hyperparameters(graph_data, input_dim, hidden_dim, output_dim):
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    best_accuracy = 0
    best_params = {}

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Training with learning rate: {lr} and batch size: {batch_size}")
            model = GNNModel(input_dim, hidden_dim, output_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # Train the model
            train(model, graph_data, optimizer, criterion, epochs=100)

            # Evaluate the model
            accuracy = evaluate(model, graph_data)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print(f"Best Accuracy: {best_accuracy} with parameters: {best_params}")

def plot_confusion_matrix(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1).numpy()
        true = data.y.numpy()  # Assuming data.y contains the true labels

    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_roc_curve(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        preds = F.softmax(out, dim=1)[:, 1].numpy()  # Probability for positive class
        true = data.y.numpy()

    fpr, tpr, thresholds = roc_curve(true, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

from sklearn.model_selection import train_test_split

def split_data(graph_data, test_size=0.2):
    num_nodes = graph_data.num_nodes
    train_indices, test_indices = train_test_split(
        range(num_nodes),
        test_size=test_size,
        stratify=graph_data.y.numpy()  
    )

    # Create a mask for the training indices
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True

    # Filter training edges
    train_edge_index = graph_data.edge_index[:, train_mask[graph_data.edge_index[0]] & train_mask[graph_data.edge_index[1]]]

    # Remap node indices for training data
    node_idx = torch.zeros(num_nodes, dtype=torch.long)
    node_idx[train_indices] = torch.arange(len(train_indices))
    train_edge_index = node_idx[train_edge_index]

    train_data = graph_data.__class__(
        x=graph_data.x[train_indices],
        edge_index=train_edge_index,
        y=graph_data.y[train_indices]
    )

    # Create a mask for the test indices
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True

    # Filter test edges (edges within test nodes only)
    test_edge_index = graph_data.edge_index[:, test_mask[graph_data.edge_index[0]] & test_mask[graph_data.edge_index[1]]]

    # Remap node indices for test data
    node_idx = torch.zeros(num_nodes, dtype=torch.long)
    node_idx[test_indices] = torch.arange(len(test_indices))
    test_edge_index = node_idx[test_edge_index]

    test_data = graph_data.__class__(
        x=graph_data.x[test_indices],
        edge_index=test_edge_index,
        y=graph_data.y[test_indices]
    )
    
    return train_data, test_data
# Call the hyperparameter tuning function

# Main function to tie everything together
def main():
    print("Loading dataset...")
    dataset = load_dataset('PKG - C-NMC 2019')
    print("Dataset loaded. Total samples:", len(dataset))
    print("Extracting features...")
    features, labels = extract_features(dataset)
    print("Features extracted. Shape:", features.shape)
    
    print("Constructing graph...")
    graph_data = construct_graph(features, labels)
    print("Graph constructed. Number of nodes:", graph_data.num_nodes)
    # After creating your graph_data
    print("Max edge index:", graph_data.edge_index.max().item())
    assert graph_data.edge_index.max() < graph_data.num_nodes, "Error: edge_index contains an invalid node reference."
    if graph_data.edge_index.max() >= graph_data.num_nodes:
        print("Error: edge index contains invalid node reference.")
    print(f'Graph created with {len(graph_data.x)} nodes and {graph_data.edge_index.size(1)} edges.')
    edge_index = graph_data.edge_index
    valid_mask = (edge_index < graph_data.num_nodes).all(dim=0)
    graph_data.edge_index = edge_index[:, valid_mask]

    print(f"Filtered edge_index. Remaining valid edges: {graph_data.edge_index.size(1)}")
    # Here you can proceed to define and train your GNN model using graph_data
    # Define model parameters
    input_dim = graph_data.x.shape[1]
    hidden_dim = 64
    output_dim = len(np.unique(labels))  # Assuming labels are available
    train_data, test_data = split_data(graph_data)
    print("Training data nodes:", train_data.num_nodes)
    print("Testing data nodes:", test_data.num_nodes)
    print("Training edge index shape:", train_data.edge_index.shape)
    print("Testing edge index shape:", test_data.edge_index.shape)


    # Initialize model, optimizer, and loss function
    model = GNNModel(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, graph_data, optimizer, criterion)
    eval=evaluate(model, graph_data)
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    # Evaluate on the training data
    train_accuracy = evaluate(model, train_data)
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    test_accuracy = evaluate(model, test_data)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    tune_hyperparameters(graph_data, input_dim, hidden_dim, output_dim)
    plot_confusion_matrix(model, graph_data)
    plot_roc_curve(model, graph_data)

if __name__ == "__main__":
    main()
