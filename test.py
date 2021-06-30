import torch
from data import PoemData
from config import Config
from model.lstm import Model


def generate(start_words):
    data = PoemData()
    ix2word = data.ix2word
    word2ix = data.word2ix
    my_config = Config()

    model = Model()
    model.load_state_dict(torch.load('model.pth'))
    model = model.cuda()
    model.eval()

    results = list(start_words)
    start_words_len = len(start_words)
    # 最开始的隐状态初始为0矩阵
    hidden = torch.zeros((2, my_config.LSTM_layers * 1, 1, my_config.hidden_dim), dtype=torch.float)
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.cuda()
    hidden=hidden.cuda()
    with torch.no_grad():
        for i in range(my_config.max_gen_len):
            output, hidden = model(input, hidden)
            # 读取输入的第一句
            if i < start_words_len:
                w = results[i]
                input = input.data.new([word2ix[w]]).view(1, 1)
            # 生成后面的句子
            else:
                top_index = output.data[0].topk(1)[1][0].item()
                w = ix2word[top_index]
                results.append(w)
                input = input.data.new([top_index]).view(1, 1)

            if w == '<EOP>' or (w == '<START>' and results[-2] == '。'):
                del results[-1]
                break

    return results

def generate_poem(start_words):
    results = generate(start_words)
    print(' '.join(i for i in results))

if __name__ == '__main__':
    generate_poem('蛙声')
    generate_poem('故乡')
    generate_poem('国科')
    generate_poem('风')
