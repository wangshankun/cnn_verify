# -*- coding: UTF-8 -*-
#https://blog.csdn.net/pengjian444/article/details/71075544
import numpy as np

def cal_rosenbrock(x1, x2):
    """
    例子的损失函数为香蕉函数,这是一个有最优解的函数,最小值为 f(1,1) = 0
    计算rosenbrock函数的值
    :param x1:
    :param x2:
    :return:
    """
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2


def cal_rosenbrock_prax1(x1, x2):
    """
    对x1求偏导
    """
    return -2 + 2 * x1 - 400 * (x2 - x1 ** 2) * x1

def cal_rosenbrock_prax2(x1, x2):
    """
    对x2求偏导
    """
    return 200 * (x2 - x1 ** 2)

def for_rosenbrock_func(max_iter_count=100000, step_size=0.001):
    pre_x = np.zeros((2,), dtype=np.float32)
    loss = 10
    iter_count = 0
    while loss > 0.0001 and iter_count < max_iter_count:
        error = np.zeros((2,), dtype=np.float32)
        error[0] = cal_rosenbrock_prax1(pre_x[0], pre_x[1])
        error[1] = cal_rosenbrock_prax2(pre_x[0], pre_x[1])

        for j in range(2):
            pre_x[j] -= step_size * error[j]

        loss = cal_rosenbrock(pre_x[0], pre_x[1])  # 最小值为0

        print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return pre_x

if __name__ == '__main__':
    w = for_rosenbrock_func()  
    print(w)
