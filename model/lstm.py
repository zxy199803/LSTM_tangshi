import torch
import torch.nn as nn
from config import Config

my_config = Config()


class Model(nn.Module):
    def __init__(self, vocab_size=my_config.vocab_size, embedding_dim=my_config.embedding_dim,
                 hidden_dim=my_config.hidden_dim):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # vocab_size:就是ix2word这个字典的长度。
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=my_config.LSTM_layers,
                            batch_first=True, dropout=0, bidirectional=False)
        self.fc1 = nn.Linear(self.hidden_dim, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, vocab_size)

    def forward(self, input, hidden=None):
        embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(my_config.LSTM_layers * 1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(my_config.LSTM_layers * 1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))  # hidden 是h,和c 这两个隐状态
        output = torch.tanh(self.fc1(output))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output, hidden
