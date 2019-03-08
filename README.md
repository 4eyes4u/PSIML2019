# PSIML2019
This repository contains solutions and public sets for PSI:ML 2019 homework.

## Problem B (Language detection)
An ordinary problem regarding probability. Hints were given so things were much easier. We had to appply Bayes formula and everything else was mostly trivial. Calculating probabilites could make problems if they are not handled correcly (division by zero) or not normalized (conditional probabilities).
Interesting part was UTF-8 encoding and parsing files since we should take care of special symbols that are unique for this type of encoding.
## Problem C (Tetris)
Parsing tetris board and figures were probably only hard parts of this problem. It was a classic brute force problem since constraints were small.
### First subtask
Trying all rotations of all figures and simply trying to put them as low as possible got maximum score.
### Second subtask
Trying DFS from first row for all rotations of all figures got maximum score. In order to optimize validation for moves or calculating score, we should use `numpy` built-in functions such as `bitwise_xor` and `multiply`.
## Problem D (Stars)