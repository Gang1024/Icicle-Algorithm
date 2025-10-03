import numpy as np
import math
from mealpy import Optimizer
from mealpy.utils.agent import Agent


class OriginalIA(Optimizer):
    """
    The original version of: Icicle Algorithm (IA)
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs):
        """
        初始化

        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """

        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])  # 检查参数有效性
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])

        self.r_max = 0  # 最大半径 r_max
        self.linear = []  # 线性衰减因子
        self.nonlinear = []  # 非线性衰减因子
        self.ice_idx = []  # 冰个体索引
        self.water_idx = []  # 水个体索引

        self.one_dim = []  # 记录第一维数据
        self.two_dim = []  # 记录第二维数据
        self.fitness = []  # 记录适应度函数值

        self.set_parameters(["epoch", "pop_size"])  # 设置参数
        self.sort_flag = False  # 选择不需要对个体进行排序
        self.is_parallelizable = True  # 选择可以并行计算

    def initialize_variables(self):
        """
        变量初始化，主要对最大半径、线性衰减因子和非线性衰减因子进行初始化
        """

        self.r_max = math.sqrt(np.sum((self.problem.ub - self.problem.lb) ** 2)) / 2  # 最大半径
        self.linear = (self.epoch + 1 - np.array(range(1, self.epoch + 1))) / (self.epoch + 1)  # 线性衰减因子（1 -> 0）
        self.nonlinear = self.linear ** (2 * (1 - self.linear))  # 非线性衰减因子（1 -> 0）

    def initialization(self):
        """
        种群初始化，若未指定种群中个体初始值，则随机生成种群个体，并计算个体权重
        """

        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)  # 根据种群规模随机生成种群个体

        self.pop = self.recalculate_weights(self.pop)  # 计算种群个体权重

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        """
        个体初始化，若未指定个体位置，则随机生成个体位置
        """

        if solution is None:
            solution = self.problem.generate_solution(encoded=True)  # 随机生成个体位置（进行编码）

        # 初始化个体全为冰个体
        weight = 0.0  # 权重
        exist_epoch = 1  # 存在的周期
        ice_weight = exist_epoch / self.epoch  # 冰重
        state = 0  # 状态：0 -> 冰, 1 -> 水
        affected_list_idx = []  # 影响/受影响个体索引

        return Agent(solution=solution, weight=weight, ice_weight=ice_weight, exist_epoch=exist_epoch, state=state,
                     affected_list_idx=affected_list_idx)

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        修正位置，若个体位置超出边界，则重新随机生成个体位置
        """

        rd = self.generator.uniform(self.problem.lb, self.problem.ub)
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        return np.where(condition, solution, rd)

    def recalculate_weights(self, pop=None):
        """
        计算种群中个体的权重
        """

        fit_total, fit_best, fit_worst = self.get_special_fitness(pop, self.problem.minmax)  # 获取种群中适应度值总和、最优解和最差解
        for idx in range(len(pop)):
            if fit_best == fit_worst:  # 若最优解和最差解一样，则说明整个种群的权重都一样，则随机设置种群权重以增加算法的随机性
                pop[idx].weight = self.generator.uniform(0.2, 0.8)
            else:  # 若最优解和最差解不一样，则按照公式4计算权重
                pop[idx].weight = 0.0001 + (pop[idx].target.fitness - fit_worst) / (fit_best - fit_worst)
        return pop

    def recalculate_ice_weights(self, pop=None):
        """
        计算种群中冰个体的重量，即公式6中 N_ice * e_i / T 的部分
        """

        self.water_idx = []  # 记录水个体的索引
        self.ice_idx = []  # 记录冰个体的索引

        for idx in range(self.pop_size):  # 遍历种群中个体
            if self.pop[idx].state == 0:  # 更新冰索引
                self.ice_idx.append(idx)

            if self.pop[idx].state == 1:  # 更新水索引
                self.water_idx.append(idx)

        for idx in range(len(pop)):  # 计算冰个体的重量
            pop[idx].ice_weight = pop[idx].exist_epoch / (self.epoch / len(self.ice_idx))

            if pop[idx].ice_weight > 1:  # 冰个体重量限制在1以内
                pop[idx].ice_weight = 1
        return pop

    def melt(self, epoch=None):
        """
        融化阶段
        """

        self.pop = self.recalculate_ice_weights(self.pop)  # 计算冰重

        for idx in range(self.pop_size):  # 遍历种群中的冰个体
            if self.pop[idx].state == 0:
                for jdx in range(idx + 1, self.pop_size):  # 遍历种群中其他的冰个体
                    if self.pop[jdx].state == 0:
                        dis = np.linalg.norm(self.pop[idx].solution - self.pop[jdx].solution)  # 计算两冰个体距离

                        if (dis < self.r_max * self.pop[idx].ice_weight or
                                dis < self.r_max * self.pop[jdx].ice_weight):  # 判断两冰个体之间的距离是否小于冰的作用半径
                            if self.problem.minmax == "min":  # 将较差的冰个体变为水（针对最小化问题）
                                if self.pop[idx].target.fitness < self.pop[jdx].target.fitness:
                                    self.pop[idx].exist_epoch += max(0, int(self.pop[jdx].exist_epoch *
                                                                            self.generator.uniform()))  # 继承冰重
                                    self.pop[jdx].state = 1
                                    self.pop[jdx].exist_epoch = 0
                                    self.pop[jdx].ice_weight = 0.0
                                else:
                                    self.pop[jdx].exist_epoch += max(0, int(self.pop[idx].exist_epoch *
                                                                            self.generator.uniform()))  # 继承冰重
                                    self.pop[idx].state = 1
                                    self.pop[idx].exist_epoch = 0
                                    self.pop[idx].ice_weight = 0.0
                                    break

                            if self.problem.minmax == "max":  # 将较差的冰个体变为水（针对最大化问题）
                                if self.pop[idx].target.fitness > self.pop[jdx].target.fitness:
                                    self.pop[idx].exist_epoch += max(0, int(self.pop[jdx].exist_epoch *
                                                                            self.generator.uniform()))  # 继承冰重
                                    self.pop[jdx].state = 1
                                    self.pop[jdx].exist_epoch = 0
                                    self.pop[jdx].ice_weight = 0.0
                                else:
                                    self.pop[jdx].exist_epoch += max(0, int(self.pop[idx].exist_epoch *
                                                                            self.generator.uniform()))  # 继承冰重
                                    self.pop[idx].state = 1
                                    self.pop[idx].exist_epoch = 0
                                    self.pop[idx].ice_weight = 0.0
                                    break

    def find_correspondence(self, epoch=None):
        """
        寻找冰和水的对应关系
        """

        self.pop = self.recalculate_ice_weights(self.pop)  # 计算冰重

        for idx in range(self.pop_size):  # 重置影响/受影响个体索引
            self.pop[idx].affected_list_idx = []

        for idx in range(self.pop_size):  # 遍历种群中个体
            for jdx in range(idx + 1, self.pop_size):  # 遍历种群中另一个体
                if self.pop[idx].state != self.pop[jdx].state:  # 若两个体为一水一冰
                    dis = np.linalg.norm(self.pop[idx].solution - self.pop[jdx].solution)  # 计算两个体距离

                    if dis < self.r_max * max(self.pop[idx].ice_weight, self.pop[jdx].ice_weight):  # 判断两个体距离是否小于冰个体的作用半径
                        self.pop[idx].affected_list_idx.append(jdx)  # 互相添加到影响/受影响个体索引中
                        self.pop[jdx].affected_list_idx.append(idx)

    def move_water_ice(self, epoch=None):
        """
        移动阶段
        """

        self.pop = self.recalculate_weights(self.pop)  # 计算权重
        pop = [self.pop[_].copy() for _ in range(self.pop_size)]  # 备份种群

        for idx in range(self.pop_size):  # 遍历种群中个体
            pos_goal = np.zeros(self.problem.n_dims)  # 初始化目标点位置

            if self.pop[idx].state == 1:  # 若个体为水个体
                if len(self.pop[idx].affected_list_idx) >= 1:  # 若水个体在冰个体范围内
                    for i in self.pop[idx].affected_list_idx:
                        pos_goal += self.pop[i].solution

                    pos_goal /= len(self.pop[idx].affected_list_idx)  # 计算水个体所属冰个体位置的均值，将其作为目标点，对应公式16集合不为空的情况

                    if self.generator.uniform() > 0.5:  # 纯水型
                        alpha = self.generator.normal(1, 1 - np.mean([self.pop[_].ice_weight
                                                                      for _ in self.pop[idx].affected_list_idx]),
                                                      self.problem.n_dims)  # 对应公式17集合不为空的情况

                        pos_new = self.pop[idx].solution + np.multiply(alpha, pos_goal - self.pop[idx].solution)  # 对应公式18
                    else:  # 离子型
                        alpha = self.generator.normal(0, 1 - np.mean([self.pop[_].ice_weight
                                                                      for _ in self.pop[idx].affected_list_idx]),
                                                      self.problem.n_dims)  # 对应公式17集合不为空的情况

                        pos_new = pos_goal + np.multiply(alpha, self.pop[self.generator.choice(self.water_idx)].solution
                                                         - self.pop[idx].solution)  # 对应公式19

                else:  # 若水不在冰个体范围内
                    weights = np.array([self.pop[_].weight for _ in self.ice_idx])

                    if np.isnan(weights).any():
                        weights = np.nan_to_num(weights)
                    weights = weights / np.sum(weights)  # 寻找冰个体权重（适应度值越好权重越大）

                    num = self.generator.integers(1, len(self.ice_idx) + 1)
                    idx_selected = self.generator.choice(self.ice_idx, size=num, p=weights, replace=False)  # 随机选择随机个数的冰个体

                    for i in idx_selected:
                        pos_goal += self.pop[i].solution

                    pos_goal /= num  # 随机选择的冰个体位置的均值

                    if self.generator.uniform() > 0.5:  # 纯水型
                        alpha = self.generator.normal(1, 1 - np.mean([self.pop[_].ice_weight for _ in idx_selected]),
                                                      self.problem.n_dims)  # 对应随机数gamma

                        pos_new = self.pop[idx].solution + np.multiply(alpha, pos_goal - self.pop[idx].solution)  # 对应公式18
                    else:  # 离子型
                        alpha = self.generator.normal(0, 1 * (1 - np.mean([self.pop[_].ice_weight for _ in
                                                                  idx_selected])), self.problem.n_dims)  # 对应随机数mu

                        pos_new = pos_goal + np.multiply(alpha, self.pop[self.generator.choice(self.water_idx)].solution
                                                         - self.pop[idx].solution)  # 对应公式19

            else:  # 若个体为冰个体
                alpha = self.generator.normal(1, 1 - self.pop[idx].ice_weight, self.problem.n_dims)  # 对应随机数alpha和beta

                if len(self.pop[idx].affected_list_idx) >= 1:  # 若冰个体范围内有水个体
                    per_rand = self.generator.uniform(-1, 1, len(self.pop[idx].affected_list_idx))
                    per_rand = per_rand / np.sum(abs(per_rand))  # 随机比例

                    for i in range(len(self.pop[idx].affected_list_idx)):
                        pos_goal += ((self.pop[self.pop[idx].affected_list_idx[i]].solution - self.pop[idx].solution)
                                     * per_rand[i])

                    pos_new = self.pop[idx].solution + np.multiply(alpha, pos_goal)  # 对应公式10

                else:    # 若冰个体范围内没有水个体
                    weights = [self.pop[_].weight for _ in self.ice_idx]
                    if np.isnan(weights).any():
                        weights = np.nan_to_num(weights)
                    weights = weights / np.sum(weights)    # 寻找冰个体权重(适应度越好权重越大)

                    num = self.generator.integers(1, len(self.ice_idx) + 1)
                    idx_selected = self.generator.choice(self.ice_idx, size=num, p=weights, replace=False)  # 随机选择随机个数的冰个体

                    for i in idx_selected:
                        pos_goal += self.pop[i].solution

                    pos_goal /= num  # 随机选择的冰个体位置的均值

                    pos_new = self.pop[idx].solution + np.multiply(alpha, pos_goal - self.pop[idx].solution)  # 对应公式11

            pos_new = self.correct_solution(pos_new)  # 记录新位置

            pop[idx].solution = pos_new  # 形成新种群

            if self.mode not in self.AVAILABLE_MODES:  # 串行计算
                pop[idx].target = self.get_target(pos_new)  # 计算适应度值

                if self.problem.minmax == "min":  # 最小化问题
                    if pop[idx].target.fitness <= self.pop[idx].target.fitness:  # 择优选择
                        if self.pop[idx].state == 0:
                            dis = np.linalg.norm(self.pop[idx].solution - pop[idx].solution)

                            if dis < self.r_max * self.pop[idx].ice_weight:  # 对应公式14
                                pop[idx].exist_epoch += len(self.pop[idx].affected_list_idx)
                            else:
                                pop[idx].exist_epoch = 1

                        self.pop[idx] = pop[idx]
                    else:
                        if self.pop[idx].state == 0:  # 对应公式14
                            self.pop[idx].exist_epoch += len(self.pop[idx].affected_list_idx)

                if self.problem.minmax == "max":  # 最大化问题
                    if pop[idx].target.fitness >= self.pop[idx].target.fitness:  # 择优选择
                        if self.pop[idx].state == 0:
                            dis = np.linalg.norm(self.pop[idx].solution - pop[idx].solution)

                            if dis < self.r_max * self.pop[idx].ice_weight:  # 对应公式14
                                pop[idx].exist_epoch += len(self.pop[idx].affected_list_idx)
                            else:
                                pop[idx].exist_epoch = 1

                        self.pop[idx] = pop[idx]
                    else:
                        if self.pop[idx].state == 0:  # 对应公式14
                            self.pop[idx].exist_epoch += len(self.pop[idx].affected_list_idx)

        if self.mode in self.AVAILABLE_MODES:  # 并行计算
            pop = self.update_target_for_population(pop)  # 更新生成的新种群中的所有新个体的适应度值
            pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)  # 对比保留较好个体

            for idx in self.ice_idx:
                if pop[idx] == self.pop[idx]:
                    pop[idx].exist_epoch += max(1, len(self.pop[idx].affected_list_idx))
                else:
                    dis = np.linalg.norm(self.pop[idx].solution - pop[idx].solution)

                    if dis < self.r_max * self.pop[idx].ice_weight:  # 对应公式14
                        pop[idx].exist_epoch += max(1, len(self.pop[idx].affected_list_idx))
                    else:
                        pop[idx].exist_epoch = 1

            self.pop = pop

    def solidify(self, epoch=None):
        """
        凝固阶段
        """

        self.pop = self.recalculate_weights(self.pop)    # 计算权重

        if len(self.water_idx) != 0:
            weights = np.array([self.pop[_].weight for _ in self.water_idx])
            if np.isnan(weights).any():
                weights = np.nan_to_num(weights)
            weights = weights / np.sum(weights)    # 寻找水个体权重(适应度越好权重越大)

            num = self.generator.integers(0, len(self.water_idx) + 1)
            idx_selected = self.generator.choice(self.water_idx, size=num, p=weights, replace=False)  # 随机选择随机个数的水个体

            for idx in idx_selected:  # 将随机选择的水个体变为冰个体
                self.pop[idx].state = 0
                self.pop[idx].exist_epoch = 1

            self.water_idx = [_ for _ in self.water_idx if _ not in idx_selected]  # 更新水个体索引
            # self.ice_idx = [_ for _ in self.ice_idx if _ not in idx_selected]
            self.ice_idx = sorted(list(idx_selected) + list(self.ice_idx))  # 更新冰个体索引

    def drip_and_drop(self, epoch=None):
        """
        坠落阶段
        """

        for idx in self.ice_idx:
            if self.generator.uniform() < 0.5:  # 对应公式22
                self.pop[idx].exist_epoch = int(self.generator.uniform(0, 1) * self.pop[idx].exist_epoch)

                if self.pop[idx].exist_epoch <= 0:  # 存在周期限制
                    self.pop[idx].exist_epoch = 1

    def record(self):
        """
        记录所有个体一维、二维以及当前迭代种群适应度均值
        """
        self.one_dim.append([self.pop[_].solution[0] for _ in range(0, self.pop_size)])
        self.two_dim.append([self.pop[_].solution[1] for _ in range(0, self.pop_size)])
        fitness_current = [self.pop[_].target.fitness for _ in range(0, self.pop_size)]
        self.fitness.append(np.mean(np.array(fitness_current)))

    def evolve(self, epoch):
        """
        每次迭代循环运行的函数

        Args:
            epoch (int): The current iteration
        """

        self.melt(epoch)  # 融化
        self.find_correspondence(epoch)  # 寻找冰和水的对应关系
        self.move_water_ice(epoch)  # 移动
        self.solidify(epoch)  # 凝固
        self.drip_and_drop(epoch)  # 坠落

        # self.record()
