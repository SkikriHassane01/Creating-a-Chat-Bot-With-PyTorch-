import torch 
import random
from dataset import load_data
from model import load_model
from utils import encode_labels

patterns, tags, intents = load_data('./Data/intents.json')
labels, label_encoder = encode_labels(tags)
tokenizer,model, device =  load_model(num_labels=len(set(labels)))
model_state = torch.load('chatbot_model.pth')
model.load_state_dict(model_state)
model.eval()

def get_intent_response(query):
    encoding = tokenizer.encode_plus(query, add_special_tokens=True, return_tensors='pt', max_length=128, padding="max_length", truncation=True)
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask = attention_mask).logits
        predicted_probs = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(predicted_probs, dim=1)
    
    confidence_threshold = 0.9
    print(confidence)
    if confidence.item() < confidence_threshold:
        return "Sorry, I'm not confident enough to respond."
    
    predicted_tag = label_encoder.inverse_transform([predicted_class.item()])[0]
    
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            random.choice(intent['responses'])
    return "Sorry, I don't understand."
    


while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response= get_intent_response(user_input)
    print(f"Bot: {response}")