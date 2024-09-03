from typing import List
import torch
import numpy as np

from constants import *
import stl   # fundamental stl expressions and operations
import stl2vec.stl2vec as stl2vec


class STL2VecL2:
    def __init__(self):
        self.stl_cache = {}

        # not to be confused with stl2vecl2
        self.stl2vec = stl2vec.STL2Vec()

        # self.waypoints = waypoints

        phis = self._sample_expressions(n=100)
        self.train(phis)

    def train(self, stl_expressions):
        self.stl2vec.run_training(stl_expressions)

    def _sample_expressions(self, n=100):
        phis = []

        # generate n random stl expressions
        for i in range(n):
            phi = None

            # random number of waypoints ~ Uniform(0, 40)
            random_num = np.random.randint(0, 40, size=1)

            # TODO: generate gt beforehand
            waypoints = np.random.rand(8, 2) * 4 - 2

            trajectory = stl.Var("q", dim=2)

            for j in range(len(random_num)):
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