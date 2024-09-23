import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import IntentDataset, load_data
from utils import encode_labels, split_data
from model import load_model

# load the data 
patterns, tags, intents = load_data('./Data/intents.json')
labels, label_encoder = encode_labels(tags)

# split data
X_train, X_test, y_train, y_test = split_data(patterns, labels)

# load model and tokenizer
num_labels = len(set(labels))
tokenizer, model, device = load_model(num_labels)

# prepare datasets and dataloaders 
train_dataset = IntentDataset(X_train, y_train, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(60):

    model.train()
    for inputs, masks, labels in train_loader:
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=inputs, attention_mask=masks).logits
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "chatbot_model.pth")
print("Model trained and saved!")