import torch
from data import PoemData
from torch.utils.data import DataLoader
from config import Config
from model.lstm import Model
from tqdm import tqdm
import torch.nn as nn
from tensorboardX import SummaryWriter
import time

my_config = Config()
data = PoemData()
ix2word = data.ix2word
word2ix = data.word2ix
writer = SummaryWriter(log_dir=my_config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

# data示例
# data[0]
# (tensor([8291, 6731, 4770, 1787, 8118, 7577, 7066, 4817,  648, 7121, 1542, 6483,
#          7435, 7686, 2889, 1671, 5862, 1949, 7066, 2596, 4785, 3629, 1379, 2703,
#          7435, 6064, 6041, 4666, 4038, 4881, 7066, 4747, 1534,   70, 3788, 3823,
#          7435, 4907, 5567,  201, 2834, 1519, 7066,  782,  782, 2063, 2031,  846]),   #text
#  tensor([6731, 4770, 1787, 8118, 7577, 7066, 4817,  648, 7121, 1542, 6483, 7435,
#         7686, 2889, 1671, 5862, 1949, 7066, 2596, 4785, 3629, 1379, 2703, 7435,
#         6064, 6041, 4666, 4038, 4881, 7066, 4747, 1534,   70, 3788, 3823, 7435,
#         4907, 5567,  201, 2834, 1519, 7066,  782,  782, 2063, 2031,  846, 7435]))  #label
# 生成数据迭代器
poem_loader = DataLoader(data, batch_size=my_config.batch_size, shuffle=True)


def train():
    model = Model()
    model.train()
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=my_config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    total_batch = 0
    for epoch in range(my_config.epoch):
        print(epoch)
        train_loss = 0.0
        train_loader = tqdm(poem_loader)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].cuda(), data[1].cuda()
            labels = labels.view(-1)  # 拼接labels对齐outputs
            optimizer.zero_grad()  # 清除上个batch的梯度信息
            outputs, hidden = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 200 == 0:
                print('\t loss:{:.4f}'.format(loss.item()))
                writer.add_scalar("loss/train", loss.item(), total_batch)
            total_batch += 1
        scheduler.step()

    # 训练结束后保存模型
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    train()
