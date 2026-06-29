from collections import deque

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import Gemini_LLM
from iohblade.methods import LLaMEA
from iohblade.loggers import ExperimentLogger
from llamea.operator import Operator
import numpy as np
import os

class SlidingWindowUCB:
    def __init__(self, operator_ids, window=50, c=1.0):
        self.window = window
        self.c = c
        self.history = deque()
        self.counts = {op: 0 for op in operator_ids}
        self.sums = {op: 0.0 for op in operator_ids}
        self.t = 0
        self.operator_ids = operator_ids

    def _ucb_score(self, op_id):
        n = self.counts[op_id]
        if n == 0:
            return float("inf")

        mean = self.sums[op_id] / n
        bonus = self.c * np.sqrt(
            np.log(max(1, min(self.t, self.window))) / n
        )
        return mean + bonus

    def _best_operator(self):
        # Ensure each operator tried once
        for op in self.operator_ids:
            if self.counts[op] == 0:
                return op

        return max(self.operator_ids, key=self._ucb_score)

    def get_weight(self, op_id, solution):
        best = self._best_operator()
        return 1.0 if op_id == best else 0.0

    def update_weight(self, op_id, reward):
        self.t += 1
        reward = max(0.0, min(1.0, reward)) # clamp reward, only consider improvements

        if len(self.history) == self.window:
            old_op, old_reward = self.history.popleft()
            self.counts[old_op] -= 1
            self.sums[old_op] -= old_reward

        self.history.append((op_id, reward))
        self.counts[op_id] += 1
        self.sums[op_id] += reward

class FitnessDistributionSelector:
    def __init__(self, operator_ids, distributions):
        self.distributions = [
            dict(zip(operator_ids, row))
            for row in distributions
        ]

    def get_weight(self, op_id, solution):
        fitness = max(0.0, solution.fitness)

        bin_count = len(self.distributions)
        bin_idx = min(int(fitness * bin_count), bin_count - 1)

        return self.distributions[bin_idx].get(op_id, 0.0)

if __name__ == "__main__": # prevents weird restarting behaviour
    experiment_name = "opstrat-select"
    api_key = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-3.1-flash-lite"
    llm = Gemini_LLM(api_key, ai_model)
    budget = 400

    operators_config = [
        ("basic", "Refine the strategy of the selected algorithm to improve it.", 1),
        ("new", "Generate a new algorithm that is different from the algorithms you have tried before.", 1),
        ("simplify", "Refine and simplify the selected algorithm to improve it.", 1),
        ("refactor", "Generate a new algorithm using core ideas from the selected algorithm.", 1),
        ("restructure", "Modify the selected algorithm by introducing a meaningful structural change that alters its overall search behaviour.", 1),
        ("correct", "Correct any mistakes in the selected algorithm.", 1),
        ("combine2", "Combine the selected algorithms by inheriting key traits from both parents.", 2),
        ("combine3", "Combine the selected algorithms by inheriting key traits from both parents.", 3),
        ("simplify2", "Simplify and recombine the selected algorithms, preserving only the most effective ideas from each.", 2),
        ("new2", "Generate a new algorithm that is different from the selected algorithms.", 2),
    ]

    DISTR_TOP3 =  [[0.0, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3333333333333333], [0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.3333333333333333], [0.0, 0.0, 0.3333333333333333, 0.0, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.3333333333333333, 0.0], [0.3333333333333333, 0.0, 0.3333333333333333, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0], [0.3333333333333333, 0.0, 0.0, 0.0, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.3333333333333333, 0.0, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.3333333333333333, 0.0, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0]]
    DISTR_GEOMETRIC_S =  [[0.08711810876945769, 0.1472296038203835, 0.24881803045644815, 0.05154917678666135, 0.03965321291281643, 0.06701392982265976, 0.030502471471397247, 0.0234634395933825, 0.113253541400295, 0.19139848496649856], [0.08711810876945766, 0.03050247147139724, 0.14722960382038347, 0.11325354140029498, 0.02346343959338249, 0.2488180304564481, 0.03965321291281641, 0.06701392982265975, 0.051549176786661335, 0.19139848496649853], [0.113253541400295, 0.08711810876945769, 0.24881803045644815, 0.06701392982265976, 0.1472296038203835, 0.19139848496649856, 0.030502471471397247, 0.0234634395933825, 0.05154917678666135, 0.03965321291281643], [0.11325354140029498, 0.03050247147139724, 0.2488180304564481, 0.06701392982265975, 0.051549176786661335, 0.19139848496649853, 0.03965321291281641, 0.02346343959338249, 0.14722960382038347, 0.08711810876945766], [0.14722960382038347, 0.03050247147139724, 0.2488180304564481, 0.06701392982265975, 0.03965321291281641, 0.19139848496649853, 0.08711810876945766, 0.02346343959338249, 0.11325354140029498, 0.051549176786661335], [0.19139848496649856, 0.0234634395933825, 0.113253541400295, 0.08711810876945769, 0.1472296038203835, 0.24881803045644815, 0.030502471471397247, 0.03965321291281643, 0.06701392982265976, 0.05154917678666135], [0.03050247147139724, 0.02346343959338249, 0.14722960382038347, 0.03965321291281641, 0.19139848496649853, 0.2488180304564481, 0.06701392982265975, 0.11325354140029498, 0.08711810876945766, 0.051549176786661335], [0.03050247147139724, 0.02346343959338249, 0.14722960382038347, 0.03965321291281641, 0.19139848496649853, 0.2488180304564481, 0.06701392982265975, 0.11325354140029498, 0.08711810876945766, 0.051549176786661335]]
    DISTR_GEOMETRIC_L =  [[0.03128054740957967, 0.12512218963831867, 0.5004887585532747, 0.007820136852394917, 0.0039100684261974585, 0.015640273704789834, 0.0019550342130987292, 0.0009775171065493646, 0.06256109481915934, 0.25024437927663734], [0.03128054740957967, 0.0019550342130987292, 0.12512218963831867, 0.06256109481915934, 0.0009775171065493646, 0.5004887585532747, 0.0039100684261974585, 0.015640273704789834, 0.007820136852394917, 0.25024437927663734], [0.06256109481915934, 0.03128054740957967, 0.5004887585532747, 0.015640273704789834, 0.12512218963831867, 0.25024437927663734, 0.0019550342130987292, 0.0009775171065493646, 0.007820136852394917, 0.0039100684261974585], [0.06256109481915934, 0.0019550342130987292, 0.5004887585532747, 0.015640273704789834, 0.007820136852394917, 0.25024437927663734, 0.0039100684261974585, 0.0009775171065493646, 0.12512218963831867, 0.03128054740957967], [0.12512218963831867, 0.0019550342130987292, 0.5004887585532747, 0.015640273704789834, 0.0039100684261974585, 0.25024437927663734, 0.03128054740957967, 0.0009775171065493646, 0.06256109481915934, 0.007820136852394917], [0.25024437927663734, 0.0009775171065493646, 0.06256109481915934, 0.03128054740957967, 0.12512218963831867, 0.5004887585532747, 0.0019550342130987292, 0.0039100684261974585, 0.015640273704789834, 0.007820136852394917], [0.0019550342130987292, 0.0009775171065493646, 0.12512218963831867, 0.0039100684261974585, 0.25024437927663734, 0.5004887585532747, 0.015640273704789834, 0.06256109481915934, 0.03128054740957967, 0.007820136852394917], [0.0019550342130987292, 0.0009775171065493646, 0.12512218963831867, 0.0039100684261974585, 0.25024437927663734, 0.5004887585532747, 0.015640273704789834, 0.06256109481915934, 0.03128054740957967, 0.007820136852394917]]
    
    def method_random():
        operators = [
            Operator(id=i, task_message=msg, parent_count=pc)
            for i, msg, pc in operators_config
        ]
        return LLaMEA(llm, budget=budget, name="LLaMEA-random", operators=operators, n_parents=4, n_offspring=4, elitism=True)

    def method_SW_UCB(suffix, window, c):
        sw_ucb = SlidingWindowUCB(
            operator_ids=[op[0] for op in operators_config],
            window=window,
            c=c
        )
        operators = [
            Operator(id=i, task_message=msg, parent_count=pc, weight_source=sw_ucb.get_weight, update_callback=sw_ucb.update_weight)
            for i, msg, pc in operators_config
        ]
        return LLaMEA(llm, budget=budget, name=f"LLaMEA-sw-ucb{suffix}", operators=operators, n_parents=4, n_offspring=4, elitism=True)
    
    def method_distribution(name, distributions):
        selector = FitnessDistributionSelector([op[0] for op in operators_config], distributions)

        operators = [
            Operator(
                id=i,
                task_message=msg,
                parent_count=pc,
                weight_source=selector.get_weight,
            )
            for i, msg, pc in operators_config
        ]

        return LLaMEA(
            llm,
            budget=budget,
            name=name,
            operators=operators,
            n_parents=4,
            n_offspring=4,
            elitism=True,
        )

    LLaMEA_method1 = method_random()                                            # random selection
    LLaMEA_method2 = method_SW_UCB("", budget, 1.0)                             # ucb
    LLaMEA_method3 = method_SW_UCB("-s", 30, 1.2)                               # sliding window ucb (small window)
    LLaMEA_method4 = method_SW_UCB("-m", 60, 1.0)                               # sliding window ucb (medium window)
    LLaMEA_method5 = method_SW_UCB("-l", 120, 0.9)                              # sliding window ucb (large window)
    LLaMEA_method6 = method_distribution("LLaMEA-top3", DISTR_TOP3)             # top3 distribution (preprocessed)
    LLaMEA_method7 = method_distribution("LLaMEA-geometric-l", DISTR_GEOMETRIC_L)   # geometric distribution (preprocessed, large factor)
    LLaMEA_method8 = method_distribution("LLaMEA-geometric-s", DISTR_GEOMETRIC_S)   # geometric distribution (preprocessed, small factor)

    methods = [LLaMEA_method1, LLaMEA_method2, LLaMEA_method3, LLaMEA_method4, LLaMEA_method5, LLaMEA_method6, LLaMEA_method7, LLaMEA_method8]
    os.makedirs(f"results/{experiment_name}", exist_ok=True)
    logger = ExperimentLogger(f"results/{experiment_name}")
    experiment = MA_BBOB_Experiment(methods=methods, runs=5, seeds=[4,7,9,12,15], dims=[5], budget_factor=2000, budget=budget, eval_timeout=60, show_stdout=True, exp_logger=logger)
    experiment()


