from builtins import range
import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    结构化SVM损失函数，朴素实现（使用循环）。

    输入数据的维度为D，共有C个类别，我们在包含N个样本的小批量数据上操作。

    输入参数:
    - W: 形状为(D, C)的numpy数组，包含权重。
    - X: 形状为(N, D)的numpy数组，包含一个小批量数据。
    - y: 形状为(N,)的numpy数组，包含训练标签；y[i] = c 表示
      X[i]的标签为c，其中 0 <= c < C。
    - reg: (float) 正则化强度

    返回一个元组:
    - loss: 单个浮点数，表示损失值
    - gradient: 关于权重W的梯度；与W形状相同的数组
    """
    dW = np.zeros(W.shape)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

    loss /= num_train

    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # 计算损失函数的梯度并将其存储在dW中。                                         #
    # 在计算损失的同时计算导数可能更简单                                           #
    # 可以修改上面的一些代码来计算梯度                                         # 
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 循环遍历每个样本来计算梯度
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                # 对于 j != y[i] 且 margin > 0 的情况
                dW[:, j] += X[i]
                # 对正确类别 y[i] 的梯度贡献
                dW[:, y[i]] -= X[i]

    # 将梯度平均
    dW /= num_train

    # 添加正则化项的梯度
    # R(W) = reg * sum(W*W)
    # d(R(W))/dW = 2 * reg * W
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    结构化SVM损失函数，向量化实现。

    输入和输出与svm_loss_naive相同。
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    #############################################################################
    # TODO:                                                                     #
    # 实现结构化SVM损失的向量化版本，将结果存储在loss中。                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    # 1. 计算所有分数 (N, C)
    scores = X.dot(W)

    # 2. 获取每个样本的正确类别的分数 (N,)
    # np.arange(num_train) 提供了 [0, 1, ..., N-1] 作为行索引
    # y 提供了 [y[0], y[1], ..., y[N-1]] 作为列索引
    correct_class_scores = scores[np.arange(num_train), y]

    # 3. 计算 margins (N, C)
    # 我们使用 correct_class_scores[:, np.newaxis] (将其变为 (N, 1))
    # 以便它可以与 (N, C) 的 scores 矩阵正确广播
    margins = scores - correct_class_scores[:, np.newaxis] + 1.0  # delta = 1

    # 4. 将正确类别 (j == y[i]) 的 margin 设置为 0，它们不应计入损失
    margins[np.arange(num_train), y] = 0

    # 5. 应用 max(0, margin)
    margins = np.maximum(0, margins)

    # 6. 计算总损失，对所有样本和所有类别求和
    loss = np.sum(margins)

    # 7. 平均损失
    loss /= num_train

    # 8. 添加正则化损失
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # 实现SVM损失梯度的向量化版本，将结果存储在dW中。                               #
    # 提示：与其从头计算梯度，重用一些损失计算时的中间值可能更容易                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    binary_mask = (margins > 0).astype(float)

    # 2. 计算每个样本 i 有多少个 j 使得 margin > 0 (N,)
    counts = np.sum(binary_mask, axis=1)

    # 3. 在正确类别 y[i] 的位置设置 -counts[i]
    # 这代表了对于 W_{y_i} 的梯度贡献
    binary_mask[np.arange(num_train), y] = -counts

    # 4. 计算梯度 dW = X^T * binary_mask
    # (D, N) x (N, C) -> (D, C)
    dW = X.T.dot(binary_mask)

    # 5. 平均梯度
    dW /= num_train

    # 6. 添加正则化梯度
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

