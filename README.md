# golfdqn
This is a Keras implementation of a basic reinforcement deep Q learning system. Adapted from https://gist.github.com/EderSantana/c7222daa328f0e885093, golfdqn was built for educational purposes. Because of its purpose, the scripts have been adapted to make input variable changes easier. The system is built to play a game with resemblance to golf. The game is simple, move the white pixel (golf ball) into the black pixel (hole) in the fewest strokes possible. There are 8 possible actions, four directions and two clubs. One club travels 5 pixels with random directional error up to one pixel. The other club, the putter travels one pixel.

![alt text](https://raw.githubusercontent.com/deanjohnr/golfdqn/master/output_HTLsHF.gif "Game GIF")

## Required Packages
* Keras with Theano or Tensorflow backend
* Numpy
* Matplotlib (optional notebook view)

## Execution - Command Line
Training - creates model.json and model.h5 for use in testing
```
python train.py
```
Testing - plays 18 holes, saves each state as an image in /gifImages, and prints the final score
```
python test.py
```
## Execution - Jupyter Notbook
PlayNotebook.ipynb provides an easy way to play around with the various game parameters and learning agents. The cells are split into train and test. Test displays the game at the end of the notebook.
