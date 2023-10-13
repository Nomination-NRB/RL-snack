import numpy as np
import config.config as basic_config
# 贪吃蛇初始长度
PLAYERSIZE = basic_config.PLAYERSIZE


class SnakeClass:
    def __init__(self, gridsize):
        self.position = np.array([gridsize // 2, gridsize // 2]).astype('float')
        self.direction = np.array([1., 0.])
        self.prevpos = [np.array([gridsize // 2, gridsize // 2]).astype('float')]
        self.gridsize = gridsize
        self.len = PLAYERSIZE

    def __len__(self):
        return self.len

    def move(self):
        """
        更新贪吃蛇位置：
            更新头部位置
            记录贪吃蛇的轨迹
        """
        self.position += self.direction
        self.prevpos.append(self.position.copy())
        self.prevpos = self.prevpos[-self.len - 1:]

    def checkdead(self, pos):
        # 贪吃蛇是否触碰游戏边界
        if pos[0] <= -1 or pos[0] >= self.gridsize:
            return True
        elif pos[1] <= -1 or pos[1] >= self.gridsize:
            return True
        # 贪吃蛇头部是否碰撞自己
        elif list(pos) in [list(item) for item in self.prevpos[:-1]]:
            return True
        else:
            return False

    def getproximity(self):
        L = self.position - np.array([1, 0])
        R = self.position + np.array([1, 0])
        U = self.position - np.array([0, 1])
        D = self.position + np.array([0, 1])
        # 四个可能的新位置
        possdirections = [L, R, U, D]
        # 检查四个方向移动是否会导致贪吃蛇死亡，死亡返回1，否则返回0
        proximity = [int(self.checkdead(x)) for x in possdirections]
        return proximity

    def showState(self):
        print('Snake:')
        print('Position:', self.position)
        print('Direction:', self.direction)
        print('Previous Positions:', self.prevpos)
        print('Length:', len(self))
        print('\n')


class AppleClass:
    def __init__(self, gridsize):
        self.position = np.random.randint(1, gridsize, 2)
        self.score = 0
        self.gridsize = gridsize

    def eaten(self):
        self.position = np.random.randint(1, self.gridsize, 2)
        self.score += 1

    def showState(self):
        print('Apple:')
        print('Position:', self.position)
        print('Score:', self.score)
        print('\n')


class GameEnvironment:
    def __init__(self, gridsize, nothing, dead, apple):
        self.snake = SnakeClass(gridsize)
        self.apple = AppleClass(gridsize)
        self.game_over = False
        self.gridsize = gridsize
        self.reward_nothing = nothing
        self.reward_dead = dead
        self.reward_apple = apple
        self.time_since_apple = 0
        self.player_moves = {
            'L': np.array([-1., 0.]),
            'R': np.array([1., 0.]),
            'U': np.array([0., -1.]),
            'D': np.array([0., 1.])
        }

    def resetgame(self):
        self.snake.position = np.random.randint(1, self.gridsize, 2).astype('float')
        self.apple.position = np.random.randint(1, self.gridsize, 2).astype('float')
        self.snake.prevpos = [self.snake.position.copy().astype('float')]
        self.apple.score = 0
        self.snake.len = PLAYERSIZE
        self.game_over = False

    def get_boardstate(self):
        return [self.snake.position, self.snake.direction, self.snake.prevpos, self.apple.position, self.apple.score,
                self.game_over]

    def update_boardstate(self, move):
        reward = self.reward_nothing
        Done = False
        # 0:Left 1:Right 2:Up 3:Down
        # 如果方向要向左，且当前方向不为右，则更新方向为左（避免碰到自己）
        if move == 0:
            if not (self.snake.direction == self.player_moves['R']).all():
                self.snake.direction = self.player_moves['L']
        if move == 1:
            if not (self.snake.direction == self.player_moves['L']).all():
                self.snake.direction = self.player_moves['R']
        if move == 2:
            if not (self.snake.direction == self.player_moves['D']).all():
                self.snake.direction = self.player_moves['U']
        if move == 3:
            if not (self.snake.direction == self.player_moves['U']).all():
                self.snake.direction = self.player_moves['D']

        self.snake.move()
        self.time_since_apple += 1
        # 经过了100步没有吃到苹果则结束游戏，避免原地转圈
        if self.time_since_apple == 100:
            self.game_over = True
            reward = self.reward_dead
            self.time_since_apple = 0
            Done = True

        if self.snake.checkdead(self.snake.position):
            self.game_over = True
            reward = self.reward_dead
            self.time_since_apple = 0
            Done = True
        elif (self.snake.position == self.apple.position).all():
            self.apple.eaten()
            self.snake.len += 1
            self.time_since_apple = 0
            reward = self.reward_apple

        len_of_snake = len(self.snake)
        return reward, Done, len_of_snake

    def showState(self):
        print('Game Environment:')
        print('Snake:')
        print('Position:', self.snake.position)
        print('Direction:', self.snake.direction)
        print('Previous Positions:', self.snake.prevpos)
        print('Length:', len(self.snake))
        print('Apple:')
        print('Position:', self.apple.position)
        print('Score:', self.apple.score)
        print('Game Over:', self.game_over)
        print('\n')


if __name__ == '__main__':
    gridsize = 10

    snake = SnakeClass(gridsize)
    snake.showState()
    snake.move()
    snake.showState()

    apple = AppleClass(gridsize)
    apple.showState()
    apple.eaten()
    apple.showState()

    env = GameEnvironment(gridsize, -0.1, -1, 1)
    env.showState()
    env.resetgame()
    env.update_boardstate(1)
    env.showState()
