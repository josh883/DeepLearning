#hw6: GNN graph neural network
#name: joshau Wilson
#date: 11-16-26

# GenAI Acknowledgment: Used Gemini (2026) to assist with PyTorch Geometric architecture setup, training loop structure, and debugging.
import torch
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#-----------------
# part1
# _________________

print("-- part 1: data loading and Explorations --")

# 1. firstly we will load the protein dataset
#TUDataset is a built in dataset class in torch_geometric.datasets
#root ='data/TUDataset' specifies the local folder path where the dataset files will be stored
#name='PROTEINS' tells the loader which specific dataset to fetch
dataset = TUDataset(root='data/TUDataset', name='PROTEINS')


# 2. retrieve and display basic dataset information
print(f"dataset: {dataset}")
print(f"total number of graphs: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Number of node features: {dataset.num_features}\n")

# 3. inspect the first grpah object: dataset[0]
first_graph = dataset[0]
print(f"----First graph object: {first_graph}")
print(f"--Node feature matrix x:\n{first_graph.x}")
print(f"--Node feature matrix x shape: {first_graph.x.shape}\n")

print(f"--Edge index matrix edge_index:\n{first_graph.edge_index}")
print(f"--Edge index matrix edge_index shape: {first_graph.edge_index.shape}\n")

print(f"---Graph label y: {first_graph.y}\n")

# 4. compute and report the average number of nodes per graph
#total_nodes = sum(data.num_nodes for data in dataset) uses a generator expression to loop through every individual graph data in the dataset
#For each graph it accesses its .num_nodes property and adds them all together into a singe integer
# avg_node takes the average
# 
total_nodes = sum(data.num_nodes for data in dataset)
avg_nodes= total_nodes / len(dataset)
print(f" ------Average number of nodes per graph: {avg_nodes:.2f}\n")

#-----------------
# part 2: data pre processing
#_________________

#RAMDOM_SEED sets a fixed value for our split in the dataset, so the random split is reproducible
# indices = list creates a python list of integers from 0 to 1112 representing the 1113 graphs in PROTEINS
# spliting index number numbers rather than the raw graph objects makes it easy to slice the dataset
# lables =[] extracts the target class label (0 or 1) for every graph in the dataset and stores them in a list
# calling  .item converts each singel -element Pytorch tensor data.y into a plain python int
print("--- part 2 : Data Pre processing ---")
RANDOM_SEED=0
indices = list(range(len(dataset)))
labels = [data.y.item() for data in dataset]

# 1. reserve 20% for testing, keeping 80% for train + validation
train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=labels
)
# 2. split the remaining 80% into 70% train and 10% validation (0.10/0.80=0.125)
train_val_labels = [labels[i] for i in train_val_idx]
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.125,
    random_state=RANDOM_SEED,
    stratify=train_val_labels
)
# Create pytorch geometric dataset subsets
train_dataset = dataset[train_idx]
val_dataset = dataset[val_idx]
test_dataset = dataset[test_idx]

print(f"Random seed: {RANDOM_SEED}")
print(f"Training set size:   {len(train_dataset)} graphs ({len(train_dataset)/len(dataset):.1%})")
print(f"Validation set size: {len(val_dataset)} graphs ({len(val_dataset)/len(dataset):.1%})")
print(f"Test set size:       {len(test_dataset)} graphs ({len(test_dataset)/len(dataset):.1%})")
print(f"Total graphs check:  {len(train_dataset) + len(val_dataset) + len(test_dataset)}\n")

# ------------------------------
# part 3 : model building and training
#_____________________________________


#set seed and device from reproducibility
#torch.manual_seed() Sets the random number generator seed inside pytorch to RANDOM_SEED(0)
#THIS ensures that random operations- such as initializing NN weights or shuffling/spitting minibatches produce the exact same results
#device = torch.device checks whether an nvidia gpu with cuda support is available on your computer, if one is
#it assigns one to the device if not it defaults to cpu
#prints which hardwar pytorch is using
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}\n")

# 3.1 prepare mini batches
# batch size defines the batch size obviously
# trainloader creates a training loader, shuffle=true  randomly re-orders the training graphs before every epoch
#so the model doesn't learn patterns from the order of the dataset
# val loader creates the validation loader, shuffle=false keeps the order consistent because no weight updates or gradient step happen during validation
#test loader creates the test loader also shuffle=false for consistent final evaluation
#data loader is a special block diagonal batching trick
#it merges 32 separate graphs into a single large disconnected ggraph batch object
# it generates a batch tensor so the gnn pooling layer knows how to aggregate
BATCH_SIZE =32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Inspect one mini-batch to verify shapes
sample_batch = next(iter(train_loader))
print(f"Sample mini-batch: {sample_batch}")
print(f"  batch.x shape:          {sample_batch.x.shape}")
print(f"  batch.edge_index shape: {sample_batch.edge_index.shape}")
print(f"  batch.batch shape:      {sample_batch.batch.shape}")
print(f"  batch.y shape:          {sample_batch.y.shape}\n")

#below is our model
# 3.2 Model 1: Graph Convolutional Network (GCN)
class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


# 3.3 Model 2: Graph Attention Network (GAT)
class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.dropout = dropout
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


# Helper function for training and validation loop
def train_and_validate(model, model_name, epochs=100, lr=0.01):
    print(f"Training {model_name}...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = -1
    best_weights = None

    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                total_val_loss += loss.item() * batch.num_graphs

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        # Retain checkpoint with lowest validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print(f"--> {model_name} Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}\n")
    return best_weights, best_epoch, best_val_loss, train_losses, val_losses


# Instantiate and train GCN
torch.manual_seed(RANDOM_SEED)
gcn_model = GCNClassifier(
    in_channels=dataset.num_features,
    hidden_channels=64,
    num_classes=dataset.num_classes
)
gcn_best_weights, gcn_best_epoch, gcn_best_val_loss, gcn_train_losses, gcn_val_losses = train_and_validate(
    gcn_model, "GCNClassifier", epochs=100, lr=0.01
)

# Instantiate and train GAT
torch.manual_seed(RANDOM_SEED)
gat_model = GATClassifier(
    in_channels=dataset.num_features,
    hidden_channels=64,
    num_classes=dataset.num_classes,
    heads=4
)
gat_best_weights, gat_best_epoch, gat_best_val_loss, gat_train_losses, gat_val_losses = train_and_validate(
    gat_model, "GATClassifier", epochs=100, lr=0.01
)


# ==========================================
# Part 4: Model Evaluation
# ==========================================
print("--- Part 4: Model Evaluation ---")

def evaluate_on_test(model, best_weights, loader):
    model.load_state_dict(best_weights)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec, f1

gcn_acc, gcn_prec, gcn_rec, gcn_f1 = evaluate_on_test(gcn_model, gcn_best_weights, test_loader)
gat_acc, gat_prec, gat_rec, gat_f1 = evaluate_on_test(gat_model, gat_best_weights, test_loader)

print(f"GCN Test Metrics (Checkpoint Epoch {gcn_best_epoch}):")
print(f"  Accuracy:  {gcn_acc:.4f}")
print(f"  Precision: {gcn_prec:.4f}")
print(f"  Recall:    {gcn_rec:.4f}")
print(f"  F1-score:  {gcn_f1:.4f}\n")

print(f"GAT Test Metrics (Checkpoint Epoch {gat_best_epoch}):")
print(f"  Accuracy:  {gat_acc:.4f}")
print(f"  Precision: {gat_prec:.4f}")
print(f"  Recall:    {gat_rec:.4f}")
print(f"  F1-score:  {gat_f1:.4f}\n")


# ==========================================
# Part 5: Plot Training & Validation Loss Curves
# ==========================================
print("--- Part 5: Saving Loss Curves Plot ---")

epochs_range = range(1, 101)
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# GCN Loss Curve
axes[0].plot(epochs_range, gcn_train_losses, label="Training Loss", color="#1f77b4", linewidth=2)
axes[0].plot(epochs_range, gcn_val_losses, label="Validation Loss", color="#ff7f0e", linewidth=2)
axes[0].axvline(gcn_best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch ({gcn_best_epoch})")
axes[0].set_title("GCNClassifier Training vs. Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend()
axes[0].grid(True, linestyle=":", alpha=0.6)

# GAT Loss Curve
axes[1].plot(epochs_range, gat_train_losses, label="Training Loss", color="#1f77b4", linewidth=2)
axes[1].plot(epochs_range, gat_val_losses, label="Validation Loss", color="#ff7f0e", linewidth=2)
axes[1].axvline(gat_best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch ({gat_best_epoch})")
axes[1].set_title("GATClassifier Training vs. Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("loss_curves.png", dpi=300)
print("Saved loss curves plot to 'loss_curves.png'.")