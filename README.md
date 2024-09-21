# Creating a Chat Bot With PyTorch

## What we will do

we're going to create a chatbot framework and conversational model for a custom database that i will train it on, the custom db will be about me, like who i'm and what i can do also what is teh skills that i have, so this chat bot can be customize by anyone and integrate it in his portfolio.

## what is the first step

A chatbot framework needs a structure in which conversational intents are defined. one of the cleanest way is with JSON file, like this:

```JSON
{
    "intents": [
        {"tag": "greeting",
         "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day"],
         "responses": ["Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help?"],
         "context_set": ""
        },
        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye"],
         "responses": ["See you later, thanks for visiting", "Have a nice day", "Bye! Come back again soon."]
        },
        {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful"],
         "responses": ["Happy to help!", "Any time!", "My pleasure"]
        }
   ]
}
```

Each conversational contains:

- **Tag:** unique name
- **Patterns:** sentence patterns for our neural network text classifier
- **responses:** one will be used as a response

So let's go and build our own chatbot from scratch
