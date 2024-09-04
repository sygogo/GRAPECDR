import torch
import torch.nn as nn
import torch.nn.functional as F




class ExternalMemoryNetwork(nn.Module):
    def __init__(self, input_size, mem_size, hidden_size):
        super(ExternalMemoryNetwork, self).__init__()
        self.input_size = input_size
        self.mem_size = mem_size
        self.hidden_size = hidden_size

        # 外部存储器
        self.values = nn.Parameter(torch.randn(mem_size, input_size))
        nn.init.xavier_uniform_(self.values.data)

        # 写入机制参数
        self.W_erase = nn.Linear(input_size, input_size)
        self.W_add = nn.Linear(input_size, input_size)

    def forward(self, mem_idx, input):
        # 生成查询
        att_score = torch.mm(input, self.values.permute(1, 0))

        # 计算注意力权重
        attn_weights = F.softmax(att_score, dim=1)

        # 使用注意力权重从存储器中检索值
        retrieved_values = torch.matmul(attn_weights, self.values)

        # 写入记忆,mem_idx如果全为0表示在测试阶段
        read_only = (mem_idx == 0).all().item()
        if not read_only:
            erase_weights = torch.sigmoid(self.W_erase(input))
            add_weights = torch.tanh(self.W_add(input))
            copy_memory = self.values.data.clone()
            copy_memory[mem_idx] = (1 - erase_weights) * self.values[mem_idx] + add_weights
            self.values.data.copy_(copy_memory)

        return retrieved_values
