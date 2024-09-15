from typing import List
import torch
import numpy as np

from constants import *
import stl   # fundamental stl expressions and operations
import stl2vec.stl2vec as stl2vec


class STL2VecL2:
    def __init__(self, num_action_params: int, len_traj_pred: int):
        self.stl_cache = {}

        # not to be confused with stl2vecl2
        self.stl2vec = stl2vec.STL2Vec()

        # for random trajectory generation
        self.num_action_params = num_action_params
        self.len_traj_pred = len_traj_pred

        phis = self._sample_expressions(n=100)
        self.train(phis)

    def train(self, stl_expressions):
        self.stl2vec.run_training(stl_expressions)

    def _sample_expressions(self, n=100):
        phis = []

        # TODO: build it in for now
        action_stats = {
            "min": [-2.5, -4], # [min_dx, min_dy]
            "max": [5, 4],  # [max_dx, max_dy]
        }

        x_min = action_stats["min"][0]
        x_max = action_stats["max"][0]
        y_min = action_stats["min"][1]
        y_max = action_stats["max"][1]

        # generate n random stl expressions
        for i in range(n):
            phi = None

            # random number of waypoints ~ Uniform(1, 40)
            # TODO: see if this is truly random, at least not in pdb
            random_num = np.random.randint(1, 40, size=1).item()

            print(f"num waypoints: {random_num}")

            # TODO: possibly generate gt beforehand
            if self.num_action_params == 2:
                X = torch.rand((random_num, self.len_traj_pred, 1), requires_grad=False) * (x_max - x_min) + x_min
                Y = torch.rand((random_num, self.len_traj_pred, 1), requires_grad=False) * (y_max - y_min) + y_min
                waypoints = torch.cat((X,Y), dim=2)
            else:
                raise ValueError("learning angle, cannot generate waypoints.")

            trajectory = stl.Var("q", dim=2)

            for j in range(random_num):
                # attach the right child to the left subtree, for every waypoint
                visit_landmark = stl.DistComparison(trajectory, waypoints[j], SIM_THRESH[0])

                if phi is None:
                    phi = visit_landmark
                else:
                    phi = stl.Until(phi, visit_landmark)

            phis.append(phi)

        return phis

    def set_cache(self, traj_name, stl):
        self.stl_cache[traj_name] = stl

    def get_cache(self):
        return self.stl_cache

    def create_stl_landmarks(self, waypoints: List) -> torch.Tensor:
        """
        Recursively generates an STL expression with the height of the parse tree being the length of the trajectory.
        The expression is defined as the Euclidean similarity between the input trajectory and the dataset's waypoints
        determined during the STL generation process.

        Returns:
            phi (`stlcg.Expression`):
                The STL formula for the current trajectory run.
        """

        phi = None

        trajectory = stl.Var("q", dim=2)

        for j in range(len(waypoints)):
            # attach the right child to the left subtree, for every waypoint
            visit_landmark = stl.DistComparison(trajectory, waypoints[j], SIM_THRESH[0])

            if phi is None:
                phi = visit_landmark
            else:
                phi = stl.Until(phi, visit_landmark)

        out = self.stl2vec(phi)

        return out