# 贪吃蛇初始长度
PLAYERSIZE = 0
# 贪吃蛇移动速度
PLAYERSPEED = 20
# 贪吃蛇默认移动方向
PLAYERDIRECTION = 1
# 游戏盘大小
GRIDSIZE = 15
# 格子大小
BLOCKSIZE = 20

# 经验回放池容量
MEMORYMAX = 1000
# 学习率
MYLR = 0.00001

# 总训练轮数
NUM_EPISODES = 20
# 每个episode进行学习更新的次数
NUM_UPDATES = 500
# 打印训练日志的频率，每经过PRINT_EVERY轮打印一次训练日志
PRINT_EVERY = 5
# 每轮内游戏的次数
GAMES_IN_EPISODE = 20
# 从经验回放池中采样的batch大小
BATCH_SIZE = 20


epsilon = 0.1
GAMMA = 0.9

