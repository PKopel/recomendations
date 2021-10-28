import matplotlib.pyplot as plt
from random import random
from typing import Dict, List


class Bandit:
    def __init__(self, bandit_id, arm_ids):
        self.bandit_id = bandit_id # name of the bandit to be displayed
        self.arm_ids = arm_ids # list od IDs of all the existing arms
    
    def recommend(self, size): # method should return list of the size `size` containing recommended items from arm_ids
        raise NotImplementedError()
    
    def feedback(self, arm_id, payout): # helper method - bandit gets results of its recommendation and can store the results
        raise NotImplementedError()


class Arm:
    def __init__(self, arm_id, activation_probability, payout_function):
        self.arm_id = arm_id
        self.activation_probability = activation_probability # how likely is that the arm would yield any payout
        self.payout_function = payout_function # function called to calculate payout

    def pull(self): # returns the value of the ayout function or 0, according to the activation probability
        if random() <= self.activation_probability:
            return max(self.payout_function(), 0)
        else:
            return 0


class Runner:
    def __init__(self, arms: Dict[str, Arm], bandits: List[Bandit]):
        self.arms = arms
        self.bandits = bandits
    
    # this method runs each and every bandit algorithm `runs` times, for given `epochs` in each run
    #   and returns all the results and payouts which then can be plotted by the latter method
    def simulate(self, runs, epochs, recommendation_size):
        results = {}
        for bandit in self.bandits:
            print("Simulating: %s" % bandit.bandit_id)
            results[bandit.bandit_id] = []
            for run in range(runs):
                run_results = []
                for epoch in range(epochs):
                    recommendation = bandit.recommend(recommendation_size)
                    epoch_payout = 0.0
                    for arm_id in recommendation:
                        payout = self.arms[arm_id].pull()
                        epoch_payout += payout
                        bandit.feedback(arm_id, payout)
                    run_results.append(epoch_payout)
                results[bandit.bandit_id].append(run_results)
        return results
    
    def plot_results(self, results, runs, epochs, mode='cumulative', scale='linear'):
        average = {bandit_id: [] for bandit_id in results}
        cumulative = {bandit_id: [] for bandit_id in results}
        for bandit_id in results:
            for e in range(epochs):
                epoch_results = []
                for r in range(runs):
                    epoch_results.append(results[bandit_id][r][e])
                
                avg_result = sum(epoch_results) / runs
                average[bandit_id].append(avg_result)
                if e == 0:
                    cumulative[bandit_id].append(avg_result)
                else:
                    cumulative[bandit_id].append(avg_result + cumulative[bandit_id][-1])
        
        if mode == 'cumulative':
            self.print_aggregated_results(cumulative)
            for bandit_id in cumulative:
                plt.plot(cumulative[bandit_id], label=bandit_id)
        elif mode == 'average':
            self.print_aggregated_results(average)
            for bandit_id in average:
                plt.plot(average[bandit_id], label=bandit_id)
        
        plt.yscale(scale)
        plt.legend()
        plt.show()

    def print_aggregated_results(self, aggregated):
        print("\nAggregated results:\n")
        for bandit_id, total_payout in sorted(aggregated.items(), key=lambda x: x[1][-1], reverse=True):
            print('%s: %s' % (bandit_id, total_payout[-1]))