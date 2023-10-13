import torch
import torch.nn as nn
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
import config.config as basic_config
from model import QNetwork, get_network_input
from Game import GameEnvironment
from collections import deque
from replay_buffer import ReplayMemory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_episode(num_games, board, model, memory):
    run = True
    games_played = 0
    total_reward = 0
    episode_games = 0
    len_array = []

    while run:
        state = get_network_input(board.snake, board.apple)
        state = state.to(device)
        action_0 = model(state)
        rand = np.random.uniform(0, 1)
        if rand > basic_config.epsilon:
            action = torch.argmax(action_0)
        else:
            action = np.random.randint(0, 5)

        reward, done, len_of_snake = board.update_boardstate(action)
        next_state = get_network_input(board.snake, board.apple)
        next_state = next_state.to(device)
        memory.push(state, action, reward, next_state, done)

        total_reward += reward

        episode_games += 1

        if board.game_over:
            games_played += 1
            len_array.append(len_of_snake)
            board.resetgame()

            if num_games == games_played:
                run = False

    avg_len_of_snake = np.mean(len_array)
    max_len_of_snake = np.max(len_array)
    return total_reward, avg_len_of_snake, max_len_of_snake


def learn(memory, model, optimizer, criterion):
    total_loss = 0

    for i in range(basic_config.NUM_UPDATES):
        optimizer.zero_grad()
        sample = memory.sample(basic_config.BATCH_SIZE)

        states, actions, rewards, next_states, dones = sample
        states = torch.cat([x.unsqueeze(0) for x in states], dim=0)
        states = states.to(device)
        actions = torch.LongTensor(actions)
        actions = actions.to(device)
        rewards = torch.FloatTensor(rewards)
        rewards = rewards.to(device)
        next_states = torch.cat([x.unsqueeze(0) for x in next_states])
        next_states = next_states.to(device)
        dones = torch.FloatTensor(dones)
        dones = dones.to(device)

        q_local = model.forward(states)
        next_q_value = model.forward(next_states)

        Q_expected = q_local.gather(1, actions.unsqueeze(0).transpose(0, 1)).transpose(0, 1).squeeze(0)

        Q_targets_next = torch.max(next_q_value, 1)[0] * (torch.ones(dones.size(), device=device) - dones)

        Q_targets = rewards + basic_config.GAMMA * Q_targets_next

        loss = criterion(Q_expected, Q_targets)

        total_loss += loss
        loss.backward()
        optimizer.step()

    return total_loss


def train(model, board, memory, optimizer, MSE):
    print('Training started on {}'.format(device))
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    avg_len_array = []
    avg_max_len_array = []

    time_start = time.time()
    temp_avg_len = 0  # 初始化 temp_avg_len
    for i_episode in range(basic_config.NUM_EPISODES + 1):

        total_reward, avg_len, max_len = run_episode(basic_config.GAMES_IN_EPISODE, board, model, memory)
        scores_deque.append(total_reward)
        scores_array.append(total_reward)
        avg_len_array.append(avg_len)
        avg_max_len_array.append(max_len)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        total_loss = learn(memory, model, optimizer, MSE)

        dt = int(time.time() - time_start)

        if i_episode % basic_config.PRINT_EVERY == 0 and i_episode > 0:
            print(
                'Ep: {:6}, Loss: {:.4f},  Reward in {}局游戏: {:.2f},  Avg.Len/{}局游戏: {:.2f},  Max.Len/{}局游戏:  {:.2f}  Time: {'
                ':02}:{:02}:{:02} '.format(i_episode, total_loss, basic_config.GAMES_IN_EPISODE, total_reward,
                                           basic_config.GAMES_IN_EPISODE, avg_len, basic_config.GAMES_IN_EPISODE,
                                           max_len, dt // 3600, dt % 3600 // 60, dt % 60))

        memory.truncate()

        if i_episode > 0 and avg_len > temp_avg_len:
            torch.save(model.state_dict(), 'dir_chk/Snake_{}'.format(i_episode))
            temp_avg_len = avg_len  # 更新 temp_avg_len

    return scores_array, avg_scores_array, avg_len_array, avg_max_len_array


def plot_scores(scores, avg_scores, avg_len_of_snake, max_len_of_snake):
    # 绘制得分和平均得分的折线图
    plt.figure()
    plt.plot(np.arange(1, len(scores) + 1), scores, label="Reward")
    plt.plot(np.arange(1, len(avg_scores) + 1), avg_scores, label="Avg Reward")
    plt.legend()
    plt.ylabel('Reward')
    plt.xlabel('Episodes #')
    plt.show()

    # 绘制平均蛇长度和最大蛇长度的折线图
    plt.figure()
    plt.plot(np.arange(1, len(avg_len_of_snake) + 1), avg_len_of_snake, label="Avg Len of Snake")
    plt.plot(np.arange(1, len(max_len_of_snake) + 1), max_len_of_snake, label="Max Len of Snake")
    plt.legend()
    plt.ylabel('Length of Snake')
    plt.xlabel('Episodes #')
    plt.show()

    # 绘制最大蛇长度的直方图
    plt.figure()
    sns.histplot(max_len_of_snake, bins=45, kde=True, color='green')
    plt.xlabel('Max Lengths')
    plt.ylabel('Probability')
    plt.title('Histogram of Max Lengths')
    plt.grid(True)
    plt.show()


def drawScores(scores, sample_interval):
    # 绘制得分和平均得分的折线图
    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(1, len(scores) + 1, sample_interval), scores[::sample_interval], label="Reward")
    plt.plot(np.arange(1, len(avg_scores) + 1, sample_interval), avg_scores[::sample_interval], label="Avg Reward")
    plt.legend()
    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    plt.savefig('./outputImage/drawScores.png')


def drawAvgAndMaxLen(avg_len_of_snake, max_len_of_snake, sample_interval):
    # 绘制平均蛇长度和最大蛇长度的折线图
    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(1, len(avg_len_of_snake) + 1, sample_interval), avg_len_of_snake[::sample_interval],
             label="Avg Len of Snake")
    plt.plot(np.arange(1, len(max_len_of_snake) + 1, sample_interval), max_len_of_snake[::sample_interval],
             label="Max Len of Snake")
    plt.legend()
    plt.ylabel('Length of Snake')
    plt.xlabel('Episodes')
    plt.savefig('./outputImage/drawAvgAndMaxLen.png')


def drawMaxHist(max_len_of_snake):
    # 绘制最大蛇长度的直方图
    plt.figure(figsize=(80, 30))
    sns.histplot(max_len_of_snake, bins=45, kde=True, color='green')
    plt.xlabel('Max Lengths')
    plt.ylabel('Probability')
    plt.title('Histogram of Max Lengths')
    plt.grid(True)
    plt.savefig('./outputImage/drawMaxHist.png')


if __name__ == "__main__":
    model = QNetwork(input_dim=10, hidden_dim=20, output_dim=5).to(device)
    board = GameEnvironment(basic_config.GRIDSIZE, nothing=0, dead=-1, apple=1)
    memory = ReplayMemory(basic_config.MEMORYMAX)
    optimizer = torch.optim.Adam(model.parameters(), lr=basic_config.MYLR)
    MSE = nn.MSELoss()

    scores, avg_scores, avg_len_of_snake, max_len_of_snake = train(model, board, memory, optimizer, MSE)

    plot_scores(scores, avg_scores, avg_len_of_snake, max_len_of_snake)
    
    # 将scores, avg_scores, avg_len_of_snake, max_len_of_snake保存到文件中
    np.save('./npy/scores.npy', scores)
    np.save('./npy/avg_scores.npy', avg_scores)
    np.save('./npy/avg_len_of_snake.npy', avg_len_of_snake)
    np.save('./npy/max_len_of_snake.npy', max_len_of_snake)

    # 读取文件中的scores, avg_scores, avg_len_of_snake, max_len_of_snake，并绘制折线图
    scores = np.load('./npy/scores.npy')
    avg_scores = np.load('./npy/avg_scores.npy')
    avg_len_of_snake = np.load('./npy/avg_len_of_snake.npy')
    max_len_of_snake = np.load('./npy/max_len_of_snake.npy')
    drawScores(scores, 500)
    drawAvgAndMaxLen(avg_len_of_snake, max_len_of_snake, 500)
    drawMaxHist(max_len_of_snake)
