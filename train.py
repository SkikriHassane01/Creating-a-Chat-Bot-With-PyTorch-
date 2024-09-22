import json
from utils import tokenize, stem, bag_of_words
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
with open ('intents.json', 'r') as f:
    intents = json.load(f)
    
all_words = []
tags= []
xy = [] # this will hold both the patterns and the corresponding tag

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    
    for pattern in intent['patterns']:
        words = tokenize(pattern)
        all_words.extend(words) #we use extent because w is an array and we don't want the all_words to be an array or arrays
        xy.append((words, tag))
        
ignore_words = ['?', '!', '.', ',', '$', '*']
all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label) 
    
X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    # access dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    # access len(dataset)
    def __len__(self):
        return self.n_samples

# hyperparameters 
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1700

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device=device)

# loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        #forward
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        #backward and optimizer 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 100 == 0: # for each 100 epoch we will print ==>
       print(f"epoch {epoch + 1} / {num_epochs}, loss = {loss.item():.11f}")
       

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data,FILE)
print('Training complete. file saved to {FILE}')