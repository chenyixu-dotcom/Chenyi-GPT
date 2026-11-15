import torch 
import pandas as pd
from Trans_model import Trans_model
import re

df = pd.read_csv("vocab.csv")
vocab = df["vocab"].tolist()
vocab_to_token = {}
token_to_vocab = {}
for i in range(0,len(vocab)):
    vocab_to_token[vocab[i]] = i
    token_to_vocab[i] = vocab[i]
    
class VanillaAgent():

    def __init__(self, tempature = 1, max_length = 80, memory_length = 256):
        self.trans_model = Trans_model()
        self.trans_model.load_state_dict(
            torch.load(
                'checkpoint/transformer_model.pth',
                map_location="cpu",
                weights_only= True
            )
        )
        self.trans_model.eval()
        self.tempature = tempature
        self.memory_length = memory_length
        self.max_length = max_length
        self.memory = torch.tensor([[vocab_to_token["<START>"]]], dtype=torch.long)

    # text to list of tokens
    def tokenize(self, text):
        word_list =  re.findall(r"\w+|[^\w\s]", text)
        return [vocab_to_token.get(t, vocab_to_token["<UNK>"]) for t in word_list]

    # complete current input till the sentence end. if detect end or if next word genearte is end, 

    def _generate(self, tokens):
        with torch.no_grad():
            for i in range(self.max_length):
                logits = self.trans_model(tokens)
                next_logit = logits[:,-1,:]
                value, indices = torch.topk(next_logit/self.tempature, k=4, dim= -1)
                
                probs = torch.softmax(value, dim = -1)
                for token_id, prob in zip(indices[0].tolist(), probs[0].tolist()):
                    print(f"{token_to_vocab[token_id]!r}: {prob:.1%}")
                sampled = torch.multinomial(probs, 1)
                token_id = torch.gather(indices, -1, sampled)
                

                tokens = torch.cat([tokens, token_id], dim = 1)

                if token_to_vocab[token_id.item()] == "<END>":
                    break

            self.memory = tokens
            return self.toString(tokens)
        
    def complete(self, input):

        tokens = self.tokenize(input)
        
        if not tokens:
            raise ValueError("expect input to be non zero")

        device = next(self.trans_model.parameters()).device

        tokens = torch.tensor(
            [tokens],
            dtype=torch.long,
            device=device
    )
        return self._generate(tokens)

    # generate the next sentence, save the generate into the memory
    def next_sentence(self):
        if self.memory is None:
            print("please enter text")
            return 
        if self.memory.size(1) > self.memory_length:
            self.memory = self.memory[:,-self.memory_length:]

        temp = self.memory.size(-1)
        self._generate(self.memory)
        generated = self.memory[:, temp:]
        return self.toString(generated)

    # return current sentence 
    def toString(self, tokens):
        words = [
            token_to_vocab[token_id]
            for token_id in tokens[0].tolist()
        ]

        return " ".join(word for word in words if word != "<END>")

    def reset(self):

        device = next(self.trans_model.parameters()).device
        self.memory = torch.tensor(
            [[vocab_to_token["<START>"]]],
            dtype=torch.long,
            device=device
        )

        print("Memory cleared.")

