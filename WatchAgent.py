import pygame
import torch
import config.config as basic_config
from Game import GameEnvironment
from model import QNetwork, get_network_input


def drawboard(win, snake, apple, block_size, windowwidth, windowheight):
    win.fill((0, 0, 0))  # 填充黑色背景

    # 绘制身体渐变色
    for i, pos in enumerate(snake.prevpos[:-1]):  # 不包括最后一个位置（蛇头）
        # 计算渐变颜色
        gradient_color = (
            int(255 * (i + 1) / len(snake.prevpos)),
            int(255 * i / len(snake.prevpos)),
            0
        )
        pygame.draw.rect(win, gradient_color, (pos[0] * block_size, pos[1] * block_size, block_size, block_size))

    # 绘制圆形蛇头
    head_pos = snake.prevpos[-1]  # 蛇头位置
    head_radius = int(block_size / 2)
    head_center = (head_pos[0] * block_size + head_radius, head_pos[1] * block_size + head_radius)
    pygame.draw.circle(win, (0, 255, 0), head_center, head_radius)

    # 绘制苹果
    pygame.draw.rect(win, (255, 0, 0),
                     (apple.position[0] * block_size, apple.position[1] * block_size, block_size, block_size))

    # 绘制背景网格
    for x in range(0, windowwidth // 2, block_size):
        pygame.draw.line(win, (50, 50, 50), (x, 0), (x, windowheight))
    for y in range(0, windowheight, block_size):
        pygame.draw.line(win, (50, 50, 50), (0, y), (windowwidth // 2, y))

    # 刷新屏幕
    pygame.display.update()


def run_snake_game(model):
    gridsize = basic_config.GRIDSIZE
    speed = basic_config.PLAYERSPEED
    block_size = basic_config.BLOCKSIZE

    board = GameEnvironment(gridsize, nothing=0., dead=-10., apple=10.)
    windowwidth = gridsize * block_size * 2
    windowheight = gridsize * block_size

    pygame.init()  # pygame 初始化
    win = pygame.display.set_mode((windowwidth, windowheight))  # 设置pygame窗口
    pygame.display.set_caption("snake")
    font = pygame.font.SysFont('arial', 18)
    clock = pygame.time.Clock()

    prev_len_of_snake = 0
    runGame = True
    allRewards = 0
    while runGame:
        clock.tick(speed)

        state_0 = get_network_input(board.snake, board.apple)
        state = model(state_0)

        action = torch.argmax(state)

        reward, done, len_of_snake = board.update_boardstate(action)
        allRewards += reward
        drawboard(win, board.snake, board.apple, block_size, windowwidth, windowheight)

        lensnaketext = font.render('  len of snake: ' + str(len_of_snake), False, (255, 255, 255))
        rewardtext = font.render('  reward: ' + str(int(allRewards)), False, (255, 255, 255))
        prevlensnaketext = font.render('  len of previous snake: ' + str(prev_len_of_snake), False, (255, 255, 255))

        win.blit(lensnaketext, (windowwidth // 2, 40))
        win.blit(rewardtext, (windowwidth // 2, 80))
        win.blit(prevlensnaketext, (windowwidth // 2, 120))

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                runGame = False

        pygame.display.update()

        if board.game_over:
            prev_len_of_snake = len_of_snake
            allRewards = 0
            board.resetgame()

    pygame.quit()


def play_snake_game_user():
    gridsize = basic_config.GRIDSIZE
    speed = basic_config.PLAYERSPEED
    block_size = basic_config.BLOCKSIZE

    board = GameEnvironment(gridsize, nothing=0., dead=-10., apple=10.)
    windowwidth = gridsize * block_size * 2
    windowheight = gridsize * block_size

    pygame.init()  # pygame 初始化
    win = pygame.display.set_mode((windowwidth, windowheight))  # 设置pygame窗口
    pygame.display.set_caption("snake")
    font = pygame.font.SysFont('arial', 18)
    clock = pygame.time.Clock()

    runGame = True

    while runGame:
        clock.tick(speed)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                runGame = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                runGame = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            paused = True
            while paused:
                clock.tick(10)
                pygame.event.pump()
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        paused = False

        # 获取玩家操作
        action = None
        if keys[pygame.K_UP]:
            action = 2
        elif keys[pygame.K_DOWN]:
            action = 3
        elif keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 1

        if action is not None:
            reward, done, len_of_snake = board.update_boardstate(action)
            drawboard(win, board.snake, board.apple, block_size, windowwidth, windowheight)

            lensnaketext = font.render(' LEN OF SNAKE: ' + str(len_of_snake), False, (255, 255, 255))
            rewardtext = font.render(' REWARD: ' + str(int(reward)), False, (255, 255, 255))

            win.blit(lensnaketext, (windowwidth // 2, 40))
            win.blit(rewardtext, (windowwidth // 2, 80))

            pygame.display.update()

            if board.game_over:
                board.resetgame()

    pygame.quit()


if __name__ == '__main__':
    # 设置一个标志，如果use_ai为True，使用深度强化学习模型玩游戏，否则让玩家自己操作游戏
    use_ai = True
    if use_ai:
        # 深度强化学习模型玩游戏
        model = QNetwork(input_dim=10, hidden_dim=20, output_dim=5)
        model.load_state_dict(torch.load('./dir_chk/Snake_60000'))
        run_snake_game(model)
    else:
        # 玩家自己操作游戏
        play_snake_game_user()



