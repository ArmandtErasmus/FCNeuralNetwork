# Fully Connected Neural Network (FCNeuralNetwork) in Python

The aim of this project is to create a feed forward neural network in python that has fully connected layers. A feed forward neural network is a network in which there is no recurrent interaction between the output and input neurons. Furthermore, a neural network is said to be fully connected when every neuron in a layer is connected to every other neuron in the preceding layer.

A basic neural network architecture, with a row vector $X$ as the input layer and a row vector $Y$ as the output layer, is shown below

$X$ $\longrightarrow$ Hidden Layers $\longrightarrow$ $Y$

The row vector $X$ contains the data that we wish to feed through the neural network. In general $X\in\mathbb{R^{i}}$ so that $X=[x_{1}, x_{2}, ..., x_{i}]^{T}$ where each $x_{i}$ is a data value and can be expressed as a neuron or equivalently a node. Each of these nodes are connected to the nodes of a subsequent layer via an edge which stores a weight value. The value of each node in a subsequent layer is computed by the weighted sum of each preceding node that is connected to it. If there is only one hidden layer which connects the input layer to the output layer, then the output nodes are computed with

$y_{j}=\sum_{i}x_{i}w_{ij} + b_{j}$

where $Y=[y_{1}, y_{2}, ..., y_{j}]$, and $B=[b_{1}, b_{2}, ..., b_{j}]$. Note that $W=[w_{ij}]$ is an $i\times j$ matrix with the weights of each edge as entries of a neuron.

This can be used to train game playing agents.

