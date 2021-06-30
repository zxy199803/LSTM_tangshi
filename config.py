class Config:
    def __init__(self):
        self.data_path = 'data/tang.npz'
        self.log_path = './log'
        self.seq_len = 48  # 根据序列长度重新划分无空格数据集
        self.batch_size = 16
        self.epoch = 10

        self.hidden_dim = 1024
        self.vocab_size = 8293
        self.embedding_dim = 100
        self.LSTM_layers = 3

        self.lr = 0.001

        self.max_gen_len = 48  # 生成唐诗的最长长度
