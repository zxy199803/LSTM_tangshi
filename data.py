import numpy as np
import torch


class PoemData:

    def __init__(self):
        from config import Config
        self.config = Config()
        self.datas = np.load(self.config.data_path, allow_pickle=True)
        self.data = self.datas['data']
        self.ix2word = self.datas['ix2word'].item()
        self.word2ix = self.datas['word2ix'].item()
        self.no_space_data = self.filter_space()  # 过滤掉'</s>'的数据,'</s>'共有4065011个

    def view_data(self):

        # self.data
        # [[8292 8292 8292 ...  846 7435 8290]  # 每首诗125个字符
        #  [8292 8292 8292 ... 7878 7435 8290]
        # ...
        # [8292 8292 8292 ... 1294 7435 8290]]

        word_data = np.zeros((1, self.data.shape[1]), dtype='<U6')  # data.shape[1]=125
        row = np.random.randint(self.data.shape[0])  # 共有data.shape[0]=57580首诗 随机挑选一首诗
        for col in range(self.data.shape[1]):
            word_data[0, col] = self.ix2word[self.data[row, col]]
        print('共有诗{}首，每首诗{}字符'.format(self.data.shape[0], self.data.shape[1]))
        print('诗词示例，第{}首诗为:'.format(row + 1))
        print(word_data)
        print('开始标志', self.ix2word[8292], 'id:8292')
        print('结束标志', self.ix2word[8290], 'id:8290')

    def __getitem__(self, idx):
        text = self.no_space_data[idx * self.config.seq_len:(idx + 1) * self.config.seq_len]
        label = self.no_space_data[idx * self.config.seq_len + 1:(idx + 1) * self.config.seq_len + 1]  # 将窗口向后移动一个字符作为标签
        text = torch.from_numpy(np.array(text)).long()
        label = torch.from_numpy(np.array(label)).long()
        return text, label

    def __len__(self):
        return int(len(self.no_space_data) / self.config.seq_len)  # 无标签数据总长度/序列长度，即为序列个数

    def filter_space(self):
        # 过滤空格
        splicing_data = torch.from_numpy(self.data).view(-1).numpy()  # 把二维的tensor拼接成一维并转回numpy
        # splicing_data   Size 7197500(57580*125)  [8292 8292 8292 ... 1294 7435 8290]
        no_space_data = []
        for i in splicing_data:
            if i != 8292:
                no_space_data.append(i)
        return no_space_data


if __name__ == '__main__':
    data = PoemData()
    data.view_data()
