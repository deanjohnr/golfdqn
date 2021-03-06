# TRAIN
import sys, getopt
import json
import numpy as np
from random import randint


class Course(object):
    def __init__(self, grid_size=20, max_strokes=50, driver_dist=5, driver_error=1):
        self.grid_size = grid_size
        self.max_strokes = max_strokes
        self.driver_dist = driver_dist
        self.driver_error = driver_error
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        putter = 1 if action < 4 else 5
        direction = action % 4
        distance = 1 if action < 4 else self.driver_dist + randint(-self.driver_error,self.driver_error)
        offline = 0 if action < 4 else randint(-self.driver_error,self.driver_error)

        if direction == 0:  # right
            action_vector = [distance,offline]
        elif direction == 1: # up
            action_vector = [offline,-distance]
        elif direction == 2: # left
            action_vector = [-distance,offline]
        elif direction == 3: # down
            action_vector = [offline,distance]

        hole_row, hole_col, ball_row, ball_col, strokes, dist_delta = state[0]
        new_ball_row, new_ball_col = [ball_row + action_vector[0], ball_col + action_vector[1]]   #min(max(1, basket + action), self.grid_size-1)
        
        # check if putt hit hole
        if (putter == 1) and (new_ball_row == hole_row):
            new_ball_col = hole_col if min(new_ball_col,ball_col) <= hole_col <= min(new_ball_col,ball_col) else new_ball_col
        elif (putter == 1) and (new_ball_col == hole_col):
            new_ball_row = hole_row if min(new_ball_row,ball_row) <= hole_row <= min(new_ball_row,ball_row) else new_ball_row
        
        # check if ball is out of bounds and apply appropriate penalty
        if (self.grid_size-1 < new_ball_row) or (new_ball_row < 0) or (self.grid_size-1 < new_ball_col) or (new_ball_col < 0):
            penalty = 1
            dist_delta = (abs(ball_row-hole_row) + abs(ball_col-hole_col)) - (abs(new_ball_row-hole_row) + abs(new_ball_col-hole_col))
        else: # if ball is out of bounds don't advance ball
            penalty = 0
            dist_delta = (abs(ball_row-hole_row) + abs(ball_col-hole_col)) - (abs(new_ball_row-hole_row) + abs(new_ball_col-hole_col))
            ball_row = new_ball_row
            ball_col = new_ball_col
        
        out = np.asarray([hole_row, hole_col, ball_row, ball_col, strokes+1+penalty, dist_delta])
        out = out[np.newaxis]
        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        im_size = (1,self.grid_size,self.grid_size)
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[0, state[0], state[1]] = -1  # draw hole
        canvas[0, state[2], state[3]] = 1  # draw ball
        return canvas

    def _get_reward(self):
        hole_row, hole_col, ball_row, ball_col, strokes, dist_delta = self.state[0]
        return dist_delta

    def _is_over(self):
        hole_row, hole_col, ball_row, ball_col, strokes, dist_delta = self.state[0]
        if ((hole_row == ball_row) and (hole_col == ball_col)) or (strokes > self.max_strokes):
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        return canvas

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over, self.state[0, 4]

    def reset(self):
        hole_row = np.random.randint(2, round(self.grid_size/3,0), size=1)
        hole_col = np.random.randint(3, self.grid_size-4, size=1)
        self.state = np.asarray([hole_row, hole_col, self.grid_size-2, round(self.grid_size/2,0), 0, 0])[np.newaxis].astype(int)
        hole_row, hole_col, ball_row, ball_col, strokes, dist_delta = self.state[0]


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.2):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=32):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape
        inputs = np.zeros((min(len_memory, batch_size), env_dim[1], env_dim[2]))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

def main(argv):
    grid_size = 20
    learning_rate = 0.1
    exploration = 0.1
    epoch = 1000
    max_memory = 100
    hidden_size = 0
    try:
        opts, args = getopt.getopt(argv,"hg:l:e:n:m:d:",["gridsize=","learn=","explore=","epoch=","memory=","hidden="])
    except getopt.GetoptError:
        print 'train.py -g <gridsize> -e <explore> -n <epoch> -m <memory> -d <hidden>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'train.py -g <gridsize> -l <learn> -e <explore> -n <epoch> -m <memory> -d <hidden>'
            sys.exit(0)
        elif opt in ("-g", "--gridsize"):
            grid_size = int(arg)
        elif opt in ("-l", "--learn"):
            learning_rate = float(arg)
        elif opt in ("-e", "--explore"):
            exploration = float(arg)
        elif opt in ("-n", "--epoch"):
            epoch = int(arg)
        elif opt in ("-m", "--memory"):
            max_memory = int(arg)
        elif opt in ("-d", "--hidden"):
            hidden_size = int(arg)
    if grid_size < 20: grid_size = 20
    if learning_rate > 1: learning_rate = 0.1
    if exploration > 1 or exploration < 0: exploration = 0.1
    if epoch < 1: epoch = 500
    if max_memory < 1: max_memory = 100
    if hidden_size <= 0: hidden_size = 2*grid_size
    return grid_size, learning_rate, exploration, epoch, max_memory, hidden_size

if __name__ == "__main__":
    # parameters
    num_actions = 8
    driver_dist = 5
    driver_error = 1
    max_strokes = 50
    batch_size = 32
    grid_size, learning_rate, exploration, epoch, max_memory, hidden_size = main(sys.argv[1:])

    # Now Import Keras
    from keras.models import Sequential
    from keras.layers.core import Dense, Flatten
    from keras.layers.convolutional import Convolution1D
    from keras.optimizers import sgd
    
    # Initialize Neural Network Agent
    model = Sequential()
    model.add(Convolution1D(grid_size, 3, input_shape=(grid_size, grid_size), activation='relu'))
    model.add(Flatten())
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=learning_rate), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    #model.load_weights("model.h5")

    # Define environment/game
    env = Course(grid_size=grid_size, max_strokes=max_strokes, driver_dist=driver_dist, driver_error=driver_error)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= exploration:
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over, strokes = env.act(action)
            if (strokes <= max_strokes) and (game_over):
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            newloss = model.train_on_batch(inputs, targets)

            loss += newloss
        print("Epoch {:03d}/{} | Loss {:.4f} | Win count {} | Strokes {}".format(e, epoch, loss, win_cnt, strokes))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)