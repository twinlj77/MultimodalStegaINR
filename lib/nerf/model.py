import torch
import torch.nn as nn

#定义一个小模型类
class SmallModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=256):
        super().__init__()
        #第一个块
        self.block1_1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU())
        self.block1_4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block1_5 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block1_8 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        #第二个块
        self.block2_1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU())
        self.block2_4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block2_5 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block2_8 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim + 1))
        # 第三和第四个块
        self.block3_3 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), nn.ReLU())
        self.block4_3 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid())

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()
    # 定义位置编码函数
    def positional_encoding(self, x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)
    # 前向传播函数
    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction)

        h = self.block1_8(self.block1_5(self.block1_4(self.block1_1(emb_x))))
        tmp = self.block2_8(self.block2_5(self.block2_4(self.block2_1(torch.cat((h, emb_x), dim=1)))))

        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3_3(torch.cat((h, emb_d), dim=1))
        c = self.block4_3(h)
        return c, sigma

# 定义一个大型模型类
class BigModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=256):
        super().__init__()

        self.block1_1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU())

        self.block1_2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block1_3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.block1_4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block1_5 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.block1_6 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block1_7 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.block1_8 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.block2_1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU())

        self.block2_2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block2_3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.block2_4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block2_5 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.block2_6 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block2_7 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block2_8 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim + 1))

        self.block3_1 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, embedding_dim_direction * 6 + hidden_dim + 3), nn.ReLU())
        self.block3_2 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, embedding_dim_direction * 6 + hidden_dim + 3), nn.ReLU())
        self.block3_3 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), nn.ReLU())

        self.block4_1 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.ReLU())
        self.block4_2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.ReLU())
        self.block4_3 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid())

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    def positional_encoding(self, x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction)

        h = self.block1_8(self.block1_7(self.block1_6(self.block1_5(self.block1_4(self.block1_3(self.block1_2(self.block1_1(emb_x))))))))
        tmp = self.block2_8(self.block2_7(self.block2_6(self.block2_5(self.block2_4(self.block2_3(self.block2_2(self.block2_1(torch.cat((h, emb_x), dim=1)))))))))

        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3_3(self.block3_2(self.block3_1(torch.cat((h, emb_d), dim=1))))
        c = self.block4_3(self.block4_2(self.block4_1(h)))
        return c, sigma


