import torch
import torch.nn as nn
import pandas as pd 

df = pd.read_csv("vocab.csv")
vocab = df["vocab"].tolist()

class transformer_block(nn.Module):
    def __init__(self, hidden_dim = 32, trans_dim = 64):
        super().__init__()
        
        self.query = nn.Linear(in_features = hidden_dim, out_features=trans_dim)
        self.key = nn.Linear(in_features = hidden_dim, out_features=trans_dim)
        self.value = nn.Linear(in_features= hidden_dim, out_features=trans_dim)

        self.attentionOutput = nn.Linear(in_features= trans_dim, out_features=hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, input, future_mask):

        normalized = self.norm1(input)
        
        q = self.query(normalized)
        k = self.key(normalized)
        v = self.value(normalized)

        scores = q@k.transpose(1,-1)
        scores = scores/q.size(-1)**0.5

        scores = scores.masked_fill(future_mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_head = attention_weights @ v

        # input add semantics
        x = input + self.attentionOutput(attention_head)
        # add feed forwarded(learned information) for the next iteraction of the block
        x = x + self.feed_forward(self.norm2(x))

        return x


class Trans_model(nn.Module):
    def __init__(self, hidden_dim = 32, trans_dim = 64):
        super().__init__() 
        self.hidden_dim = hidden_dim
        self.trans_dim = trans_dim
        self.embedding_dim= 128
        self.tokens = nn.Embedding(num_embeddings=len(vocab), embedding_dim=self.embedding_dim)

        self.max_length = 256

        self.position_embedding = nn.Embedding(
            self.max_length,
            self.embedding_dim
        )

        # fully connected layer for vocab
        self.tokenFCP = nn.Linear(in_features = self.embedding_dim, out_features=hidden_dim)

        # fully connected layer for output
        self.outputFCP = nn.Linear(in_features=hidden_dim, out_features= len(vocab))
        self.block = nn.ModuleList([
            transformer_block(self.hidden_dim, self.trans_dim) for i in range(6)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
    def forward(self, input):
        # input = batch*input vocab* embedded dim
        sequence_length = input.size(1)
        if sequence_length > self.max_length:
            raise ValueError("Input exceeds max_length")
        
        positions = torch.arange(sequence_length, device=input.device)
        position_vector = self.position_embedding(positions)
        word = self.tokens(input)
        input = word + position_vector
        input = self.tokenFCP(input)

        # mask
        future_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=input.device
            ),
            diagonal=1
        )

        for block in self.block:
            input = block(input,future_mask)
        input = self.final_norm(input)
        return self.outputFCP(input)
