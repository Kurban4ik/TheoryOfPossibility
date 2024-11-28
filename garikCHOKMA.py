import functools
from math import factorial
import matplotlib.pyplot as plt
import numpy as np

args = [1 / 7, 1 / 46, 20, 0]


# Функция для просчёта вероятности отказа
@functools.lru_cache()
def Pi(lam, mu, n, m, pos=None):
    """
    :param lam: -
    :param mu: -
    :param n: amount of factory lines
    :param m: query amount
    :param pos: if None returns possibility of reject else possibility of pos position
    :return: possibility of reject
    """
    p0 = 0
    ro = lam / mu
    if m != -1:
        for i in range(n + 1):
            p0 += ro ** i / factorial(i)
        p0 = 1 / (p0 + (ro ** n / factorial(n)) * sum([(ro ** j / n ** j) for j in range(1, m + 1)]))
    else:
        q = sum([ro ** i / factorial(i) for i in range(0, n + 1)]) + ro ** (n + 1) / factorial(n) / (n - ro)
        p0 = 1 / q
    if pos == None:
        return (ro ** (n + m) / (factorial(n) * n ** m)) * p0
    return (ro ** pos / (factorial(min(n, pos)) * n ** max(0, pos - n))) * p0


def task1_2():
    # Начальные параметры для Pi (lam, mu, n, m), pos определяется отдельно
    args1 = args.copy()
    args2 = args.copy()

    # Ответы в формате вида (значения графика по икс: list, значения графика по игрек: list)
    ans_to_var_n = []
    ans_to_var_m = []
    for i in range(20): ans_to_var_n.append([]), ans_to_var_m.append([])

    # Блок просчёта вероятности отказа (0)
    for i in range(1, 20 + 1):
        p_otk_ni = []
        p_otk_mi = []
        args1[2] = i
        args2[3] = i
        for j in range(1, 20 + 1):
            args1[3] = j
            args2[2] = j
            p_otk_ni.append(Pi(*args1))
            p_otk_mi.append(Pi(*args2))
        ans_to_var_n[0].append((range(1, 20 + 1), p_otk_ni))
        ans_to_var_m[0].append((range(1, 20 + 1), p_otk_mi))
    # for i in ans_to_var_n[1]: plt.plot(*i)

    # Блок просчёта матожидания (1)
    for i in range(1, 20 + 1):
        args1[2] = i
        args2[3] = i
        ys_var_m = []
        ys_var_n = []
        for j in range(1, 21):
            args1[3] = j
            args2[2] = j
            ys_var_m.append(sum([Pi(*args1, k) * min(k, i) for k in range(i + j + 1)]))
            ys_var_n.append(sum([Pi(*args2, k) * min(k, j) for k in range(i + j + 1)]))
        ans_to_var_m[1].append((list(range(1, 21)), ys_var_m))
        ans_to_var_n[1].append((list(range(1, 21)), ys_var_n))

    # for i in ans_to_var_n[1]: plt.plot(*i)

    # Блок расчёта коэффициента загруженности операторов (2)

    for i in ans_to_var_n[1]:
        new_q = [i[1][k] / i[0][k] for k in range(len(i[1]))]
        ans_to_var_n[2].append((i[0], new_q))
    for i in range(1, len(ans_to_var_m[1]) + 1):
        new_q = np.array(ans_to_var_m[1][i - 1][1]) / i
        ans_to_var_m[2].append((ans_to_var_m[1][i - 1][0], new_q))
    # for i in ans_to_var_m[2]: plt.plot(*i)

    # Блок расчёта вероятности существования очереди (3)
    for i in range(1, 21):
        args1[2] = i
        args2[3] = i
        ys_var_n = []
        ys_var_m = []
        for j in range(1, 21):
            args1[3] = j
            args2[2] = j
            ys_var_n.append(sum([Pi(*args1, i + k) for k in range(1, j + 1)]))
            ys_var_m.append(sum([Pi(*args2, j + k) for k in range(1, i + 1)]))
        ans_to_var_m[3].append((range(1, 21), ys_var_m))
        ans_to_var_n[3].append((range(1, 21), ys_var_n))
    # for i in ans_to_var_m[3]: plt.plot(*i)

    # Блок расчёта матожидания длины очереди (4)
    for i in range(1, 21):
        args1[2] = i
        args2[3] = i
        ys_var_m = []
        ys_var_n = []
        for j in range(1, 21):
            ys_var_m.append(sum([Pi(*args1[:3], j, k + i) * k for k in range(1, j + 1)]))
            ys_var_n.append(sum([Pi(*args1[:2], j, i, j + k) * k for k in range(1, i + 1)]))
        ans_to_var_m[4].append((range(1, 21), ys_var_m))
        ans_to_var_n[4].append((range(1, 21), ys_var_n))
    # for i in ans_to_var_n[4]: plt.plot(*i)

    # Блок расчёта коэффициента занятости мест (5)
    for i in ans_to_var_m[4]:
        new_q = [i[1][k] / i[0][k] for k in range(len(i[1]))]
        ans_to_var_m[5].append((i[0], new_q))
    for i in range(1, len(ans_to_var_n[4]) + 1):
        new_q = np.array(ans_to_var_n[4][i - 1][1]) / i
        ans_to_var_n[5].append((ans_to_var_n[4][i - 1][0], new_q))
    # for i in ans_to_var_n[5]: plt.plot(*i)


def task1_3():
    ans = []
    for i in range(20): ans.append([])
    args1 = args.copy()
    args1[-1] = -1

    ro = args1[0] / args1[1]
    # Блок подсчёта числа занятых операторов (0)
    ans[0] = [[range(1, 21), [ro] * 20]]

    # Блок подсчёта коэффициента загрузки операторов (1)
    ans[1] = [[range(1, 21), [ro / i for i in range(1, 21)]]]

    # Блок подсчёта вероятности сущестсования очереди
    ans[2] = [[range(1, 21), [Pi(*args1[:2], i, -1, i) * ro / (i - ro) for i in range(1, 21)]]]

    # Блок подсчёта матожидания длины очереди в зависимости от числа операторов
    ans[3] = [[range(1, 21), [Pi(*args1[:2], i, -1, i) * ro / i / (1 - ro / i) ** 2 for i in range(1, 21)]]]

    return ans

def task1_4():


if __name__ == '__main__':
    anses = task1_3()
    for i in anses[2]: plt.plot(*i)                                     
    plt.show()
    plt.close()
