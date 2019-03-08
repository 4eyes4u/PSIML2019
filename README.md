# PSIML2019
This repository contains solutions and public sets for PSI:ML 2019 homework.

## Problem B (Language detection)
An ordinary problem regarding probability. Hints were given so things were much easier. We had to appply Bayes formula and everything else was mostly trivial.
Calculating probabilites could make problems if they are not handled correcly (division by zero) or not normalized (conditional probabilities).
Interesting part was UTF-8 encoding and parsing files since we should take care of special symbols that are unique for this type of encoding.
## Problem C (Tetris)
Parsing tetris board and figures were probably only hard parts of this problem. It was a classic brute force problem since constraints were small.
### First subtask
Trying all rotations of all figures and simply trying to put them as low as possible got maximum score.
### Second subtask
Trying DFS from first row for all rotations of all figures got maximum score.
In order to optimize validation for moves or calculating score, we should use `numpy` built-in functions such as `bitwise_xor` and `multiply`.
## Problem D (Stars)
### Idea
The most interesting problem for this year even if it was a geometry problem. The most important part in forming an idea was to realize that distortion is simply an affine mapping.
Affine mapping preserves barycenters so we should fine such one (and only one) that mapps them elementwise.
Shape recognition should be performed on original image and later just mapped in distorted one. Affine mapping is uniquely determined by three non-collinear points.
Trying out all combinations would be tremendous waste of time because it's obvious that we can apply some heuristics. First of all, the biggest triangle (measured by area) in the original image
has to be mapped into the biggest triangle in the distorted image. In order to brute force them easily, we can compute convex hull of original barycenters and try all permutations.
### Shape recognition
It was stated that private set looks almost exactly as public one so we should notice every possible pattern. Shapes are inscribed in the square with the same side length.
There can be only 64 such shapes in the image. So how to determine which shape is what?
#### Circle shapes
Circle shapes have noticeably more vertices on the convex hull since such hull is approximating a circle. If barycenter is white we definetly have a circle. In order to distinguish
flower from donut, we should notice that ray traced from origin that is parallel to x-axis can change its' color only one in case shape is a donut. Otherwise, it's a donut.
#### Cross vs. star
They are all inscribed in a 5-gon so we can't break a tie by that. But, cross has noticeably less pixels so we can use ratio of convex hull area and number of pixels.
#### Spiral
If shape is not determine yet, it's a spiral.
