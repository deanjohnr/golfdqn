# TEST # From https://gist.github.com/EderSantana/c7222daa328f0e885093
import json
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
import matplotlib.pylab as pl
from IPython import display
import time

# Make sure this grid size matches the value used fro training
grid_size = 20

with open("model.json", "r") as jfile:
    model = model_from_json(json.load(jfile))
model.load_weights("model.h5")
model.compile("sgd", "mse")

# Define environment, game
env = Course(grid_size)
c = 0
for e in range(18): # 18 holes
    loss = 0.
    env.reset()
    game_over = False
    # get initial input
    input_t = env.observe()
    plt.imshow(input_t.reshape((grid_size,)*2),
               interpolation='none', cmap='gray')
    plt.savefig("gifImages/%03d.png" % c)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    c += 1
    while not game_over:
        input_tm1 = input_t

        # get next action
        q = model.predict(input_tm1)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        input_t, reward, game_over, strokes = env.act(action)

        plt.imshow(input_t.reshape((grid_size,)*2),
                   interpolation='none', cmap='gray')
        #plt.savefig("gifImages/%03d.png" % c)
        display.display(plt.gcf())
        display.clear_output(wait=True)
        time.sleep(0.01)
        c += 1