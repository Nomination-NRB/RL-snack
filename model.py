import torch
import torch.nn as nn
import Game


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        l1 = self.relu(self.fc1(x.float()))
        l2 = self.relu(self.fc2(l1))
        l3 = self.relu(self.fc3(l2))
        l4 = self.fc4(l3)
        return l4


def get_network_input(player, apple):
    """
    获取游戏状态信息组成输入向量：
        贪吃蛇头部
        苹果的位置
        贪吃蛇头部的方向
        贪吃蛇头部周围环境
    """
    proximity = player.getproximity()
    x = torch.cat([torch.from_numpy(player.position).double(), torch.from_numpy(apple.position).double(),
                   torch.from_numpy(player.direction).double(), torch.tensor(proximity).double()])
    return x


if __name__ == '__main__':
    model = QNetwork(input_dim=10, hidden_dim=20, output_dim=5)
    print(model)
    # 创建Game环境
    env = Game.GameEnvironment(gridsize=10, nothing=-0.1, dead=-1, apple=1)

    # 重置环境,获取snake和apple对象
    env.resetgame()
    player = env.snake
    apple = env.apple

    # 测试get_network_input
    Input = get_network_input(player, apple)
    print(Input)

