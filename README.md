# Augmented_Reality_Sudoku_Solver

![video-to-gif-converter (3)](https://user-images.githubusercontent.com/58707157/85078298-6ab25580-b1c4-11ea-80ce-209a2c520727.gif)

## Overview
The project consists of 3 parts:
  1. Compute Vision - To detect and extract a sudoku from a camera input.
  2. Deep Learning - To predict whether each cell contains a number (0 to 9) or is just empty.
  3. Python Algorithm - To solve the sudoku
  
 ## Notes
  1. The backtracking algorithm to solve a sudoku is very slow, so I wrote another algorithm (check solve.py).
  2. The algorithm can solve easy, medium and hard sudokus.
  3. You can also check the code for training the model on colab here:
    https://colab.research.google.com/drive/1sAMzIrJAwyV1WjHtAjDKP5kt3twjX_IY?usp=sharing
