import os 
import sys 
import time 
import torch
import numpy as np
import pickle
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from cnn_model_class import CNN
from rnn_model_class import RNN

device = torch.device("mps:0")

#----------------- Prepare the Data -----------------#

batch_size = 256

data_folder = "data/data_train_embeddings"
train_input_path = os.path.join(data_folder, "np_train_input_80.pkl")
train_labels_path = os.path.join(data_folder, "np_train_labels.pkl")
test_input_path = os.path.join(data_folder, "np_test_input_80.pkl")
test_labels_path = os.path.join(data_folder, "np_test_labels.pkl")

with open(train_input_path, "rb") as f: 
    np_train_input = pickle.load(f)

with open(train_labels_path, "rb") as f: 
    np_train_labels = pickle.load(f)
    
# loading the test data 
with open(test_input_path, "rb") as f:
    np_test_input = pickle.load(f)
    
with open(test_labels_path, "rb") as f: 
    np_test_labels = pickle.load(f)
    
# converting the data to torch tensors
tensor_train_input = torch.from_numpy(np_train_input)
tensor_train_labels = torch.from_numpy(np_train_labels)

tensor_test_input = torch.from_numpy(np_test_input)
tensor_test_labels = torch.from_numpy(np_test_labels)

# creating the dataset
train_dataset = TensorDataset(tensor_train_input, tensor_train_labels)
test_dataset = TensorDataset(tensor_test_input, tensor_test_labels)

# print(f"train_dataset.tensors[0].shape = {train_dataset.tensors[0].shape}, train_dataset.tensors[1].shape = {train_dataset.tensors[1].shape}")
# print(f"test_dataset.tensors[0].shape = {test_dataset.tensors[0].shape}, test_dataset.tensors[1].shape = {test_dataset.tensors[1].shape}")
    
# creating the dataloader
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

# print(f"len(train_loader) = {len(train_loader)}, len(test_loader) = {len(test_loader)}")

#----------------- Model and hyperparameters -----------------#

#------------------ CNN ------------------#

# learning_rate = 0.003
# momentum = 0.9

# model = CNN(use_pretrained_embeddings = False,
#             embedding_dim=100,
#             n_filters=100,
#             n_classes=2,
#             filter_size=3,
#             output_dim=100,
#             vocab_size=10000)

# model.to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
# loss_fn = nn.CrossEntropyLoss()

# num_epochs = 20

# model_save_path = "models/cnn_train_embeddings.pth"
# plots_path = "plots/cnn_train_embeddings"
# loss_plot_save_path = os.path.join(plots_path, "loss.png")
# acc_plot_save_path = os.path.join(plots_path, "accuracy.png")

#------------------ RNN ------------------#

learning_rate = 0.001
clip = 1

model = RNN(use_pretrained_embeddings=False,
        no_layers=1,
        vocab_size=10000,
        hidden_dim=200,
        embedding_dim=100,
        output_dim=50,
        n_classes=2,
        drop_prob=0.4)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.4)
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = 0.1)
# optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 40

model_save_path = "models/rnn_train_embeddings.pth"
plots_path = "plots/rnn_train_embeddings"
loss_plot_save_path = os.path.join(plots_path, "loss.png")
acc_plot_save_path = os.path.join(plots_path, "accuracy.png")

#----------------- functions -----------------#

def train_one_epoch(clip=None):
    
    '''
    returns the train_loss and train_acc
    '''
    
    model.train()
    
    running_train_loss = 0.
    total_train_samples = 0
    correct_train_samples = 0
    
    for i, data in enumerate(train_loader):
        
        # Every data instance is an input + label pair
        input, label = data
        input = input.to(device) 
        label = label.to(device) 

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output = model(input)

        # Compute the loss and its gradients
        loss = loss_fn(output, label)
        loss.backward()

        # Adjust learning weights
        if clip is not None: 
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # Gather data and report
        running_train_loss += loss.item()
        total_train_samples += label.size(0)
        
        _, pred_label = torch.max(output, dim=1)
        correct_train_samples += (pred_label == label).sum().item()
    
    train_accuracy = correct_train_samples / total_train_samples
    train_loss = running_train_loss / len(train_loader)
    
    return train_loss, train_accuracy
    
def test_one_epoch():
    
    ''' 
    returns the test_loss and test_acc
    '''
    
    model.eval()
    
    running_test_loss = 0. 
    total_test_samples = 0
    correct_test_samples = 0
    
    with torch.no_grad():
        
        for i, data in enumerate(test_loader):
            
            # Every data instance is an input + label pair
            input, label = data
            input = input.to(device) 
            label = label.to(device) 
            
            # Make predictions for this batch
            output = model(input)

            # Compute the loss
            loss = loss_fn(output, label)

            # Gather data and report
            running_test_loss += loss.item()
            total_test_samples += label.size(0)
            
            _, pred_label = torch.max(output, dim=1)
            correct_test_samples += (pred_label == label).sum().item()
        
        test_accuracy = correct_test_samples / total_test_samples
        test_loss = running_test_loss / len(test_loader)
        
        return test_loss, test_accuracy
        
def plot_train_test(train_list, test_list, kind, save_path):
    # Create a range of epochs for the x-axis
    epochs = range(1, len(train_list) + 1)

    # Plot the train and test losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_list, label=f'Train {kind}', marker='o')
    plt.plot(epochs, test_list, label=f'Test {kind}', marker='o')

    # Add labels and a legend
    plt.xlabel('Epoch')
    plt.ylabel(f'{kind}')
    plt.title(f'Train and Test {kind} Over Epochs')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.tight_layout()

    # Save the plot 
    plt.savefig(save_path)
        
if __name__ == "__main__":
    
    for name, param in model.named_parameters():
        print(f"name = {name}, device = {param.device}, shape = {param.shape}")
    
    train_loss_list = []
    train_acc_list = []
    
    test_loss_list = []
    test_acc_list = []
    
    start_time = time.time()
    
    for epoch in tqdm(range(num_epochs)):
        
        print(f"-------------- epoch = {epoch} --------------")
        
        train_loss, train_acc = train_one_epoch(clip=clip)
        test_loss, test_acc = test_one_epoch()
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        
        print(f"train_loss = {train_loss}, train_acc = {train_acc}, test_loss = {test_loss}, test_acc = {test_acc}")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    plot_train_test(train_loss_list, test_loss_list, "Loss", loss_plot_save_path)
    plot_train_test(train_acc_list, test_acc_list, "Accuracy", acc_plot_save_path)
    
    # save the model
    model.to("cpu")
    torch.save(model.state_dict(), model_save_path)
    
    