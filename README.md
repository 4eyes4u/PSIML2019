# PSIML2019
This repository contains solutions and public sets for PSI:ML 2019 homework.

## Problem B (Language detection)
## Problem C (Tetris)
Parsing tetris board and figures were probably only hard parts of this problem. It was a classic brute force problem since constraints were small.
### First subtask
Trying all rotations of all figures and simply trying to put them as low as possible got maximum score.
### Second subtask
Trying DFS from first row for all rotations of all figures got maximum score. In order to optimize validation for moves or calculating score, one should use `numpy` built-in functions such as `bitwise_xor` and `multiply`.
## Problem D (Stars)