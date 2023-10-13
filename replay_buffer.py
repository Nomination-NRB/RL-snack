import random


class ReplayMemory(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def truncate(self):
        self.buffer = self.buffer[-self.max_size:]

    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    # 经验回放缓冲区的最大容量
    max_size = 1000
    replay_memory = ReplayMemory(max_size)

    # 添加一些经验元组到缓冲区
    for i in range(10):
        state = [random.random() for _ in range(4)]  # 用随机数代替状态
        action = random.randint(0, 3)  # 随机选择一个动作
        reward = random.random()  # 随机生成奖励
        next_state = [random.random() for _ in range(4)]  # 用随机数代替下一个状态
        done = random.choice([True, False])  # 随机生成是否结束

        replay_memory.push(state, action, reward, next_state, done)

    # 输出缓冲区中的经验个数
    print("Number of experiences in replay memory:", len(replay_memory))

    # 从缓冲区中采样一个批次的经验
    batch_size = 5
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_memory.sample(batch_size)

    # 输出采样到的经验
    print("Sampled state batch:", state_batch)
    print("Sampled action batch:", action_batch)
    print("Sampled reward batch:", reward_batch)
    print("Sampled next state batch:", next_state_batch)
    print("Sampled done batch:", done_batch)
