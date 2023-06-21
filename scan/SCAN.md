## SCAN - Semantic Clustering by adopting Nearest Neighbors

Addressing unsupervised learning problem by grouping semantically meaningful clusters with out labelled data. 


Breaks into a two steps approach.
First step can be broken into two steps

1.1 - Train CNN on a pretext task 

1.2 - Mine nearest neighbors based on feature similarity.

2 - Classify group images together then train maximize dot products after softmax.

* look like most of the gains happen in the first step


## 1.1 Representation Learning

Representation learning involves selecting a pretext task. Some pretext tasks are based on specific image transformations resulting in learned features being **covariant** to transformation. not good.

Pretext tasks used


## 1.2 Semantic Clustering

## 2.0 Fine tuning through self labeling