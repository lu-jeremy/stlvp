"""Saveri, Nenzi & Bortolussi et al. (2024) Stl2vec: Semantic and Interpretable
Vector Representation of Temporal Logic, arXiv.
"""

from typing import Optional

import torch
import torch.nn as nn
# import mlflow

from sklearn.decomposition import PCA

import stl

import sys
print(sys.path)

from stl2vec.measure import BaseMeasure
from stl2vec.kernel import STLKernel
from stl2vec.gram_matrix import GramMatrix


class STL2Vec(nn.Module):

    def __init__(
            self,
            dimension : int = 32,
            signal_dimension : int = 2,
            # cross_validate=False,
            # alpha=-2,
            # alpha_min=-6,
            # alpha_max=1,
            # cv_steps=29,
            training_run_id : Optional[str] = None,
    ):
        super().__init__()

        # the dimension (which is << the number of the training STL expressions)
        self.dimension = dimension
        self.context_size = dimension

        # TODO: Figure out these parameters and if this is a good traj measure.
        initial_state_std = 1.0
        total_variation_std = 1.0
        measure = BaseMeasure(
            sigma0=initial_state_std,
            sigma1=total_variation_std,
            q=0.1
        )

        # Kernel function based on 10000 samples from mu
        self.kernel = STLKernel(measure, samples=10, sigma2=0.44, signal_dimension=signal_dimension)

        # self.cross_validate = cross_validate
        # self.alpha = alpha
        # self.alpha_min = alpha_min
        # self.alpha_max = alpha_max
        # self.cv_steps = cv_steps
        # self.train_obs = None

        if training_run_id is None:
            # raise NotImplementedError()
            self.gram = None
            self.pca = PCA(n_components=dimension)
            self.trained = False
        # else:
        #     self._load_from_training_run(training_run_id)
        #     self.trained = True

    # def _load_from_training_run(self, training_run_id):
    #     logged_model_path = f'runs:/{training_run_id}/stl2vec'
    #     model = mlflow.pytorch.load_model(logged_model_path)
    #     self.gram = model.gram
    #     self.pca = model.pca

    def run_training(
        self,
        train_phis,
        store_robustness : bool = True,
    ):
        """
        Take a list of STL expressions...
        """

        self.gram = GramMatrix(
            self.kernel, train_phis, store_robustness=store_robustness
        )

        self.pca.fit(self.gram.gram)

        self.trained = True

    def forward(self, phis):
        assert self.trained, "stl2vec must be trained!"

        if isinstance(phis, stl.Node):
            phis: list = [phis]

        # performance of every phi in phis on the
        x = self.gram.compute_bag_kernel_vector(phis)
        out = self.pca.transform(x)

        return out

    def build_context(self, dataset, input_dict):
        context = dict()
        context['stl_expressions'] = input_dict["stl_expression"]
        return context