# Simam: A simple, parameter-free attention module for convolutional neural networks (ICML 2021)
# https://github.com/ZjjConan/SimAM/blob/master/networks/attentions/simam_module.py

import torch
import torch.nn as nn


class SimAMAttention(nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAMAttention, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


def main():
    attention_block = SimAMAttention()
    input = torch.ones([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
