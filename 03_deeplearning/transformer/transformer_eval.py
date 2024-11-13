import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Date

from  transformer_demo import tgt_vocab, loader, idx2word, Transformer
# 姑且把导包也放在这个地方吧

# 原文使用的是大小为4的beam search，这里为简单起见使用更简单的greedy贪心策略生成预测，不考虑候选，每一步选择概率最大的作为输出
# 如果不使用greedy_decoder，那么我们之前实现的model只会进行一次预测得到['i']，并不会自回归，所以我们利用编写好的Encoder-Decoder来手动实现自回归（把上一次Decoder的输出作为下一次的输入，直到预测出终止符）
