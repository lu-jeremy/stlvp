import torch
import torch.nn as nn

from torch import Tensor

import numpy as np
# import stl


class STLKernel(nn.Module):
    def __init__(
        self,
        measure,
        normalize=True,
        exp_kernel=True,
        sigma2=0.2,
        integrate_time=False,
        samples=100000,
        signal_dimension=2,
        boolean=False,
        signals=None,
    ):
        super().__init__()

        self.measure = measure
        self.exp_kernel = exp_kernel
        self.normalize = normalize
        self.sigma2 = sigma2
        self.samples = samples
        self.signal_dimension = signal_dimension
        self.integrate_time = integrate_time
        if signals is not None:
            self.signals = signals
        else:
            signals = self.measure.sample(samples=samples, varn=signal_dimension)
            self.register_buffer("signals", signals)

        self.boolean = boolean

    def forward(self, phi1, phi2):
        return self.compute_one_one(phi1, phi2)

    def compute_one_one(self, phi1, phi2):
        phis1: list = [phi1]
        phis2: list = [phi2]
        ker = self.compute_bag_bag(phis1, phis2)
        return ker[0, 0]

    def compute_bag(self, phis, return_robustness=True):
        if self.integrate_time:
            rhos, selfk, len0 = self._compute_robustness_time(phis)
            kernel_matrix = self._compute_kernel_time(
                rhos, rhos, selfk, selfk, len0, len0
            )
        else:
            rhos, selfk = self._compute_robustness_no_time(phis)
            kernel_matrix = self._compute_kernel_no_time(rhos, rhos, selfk, selfk)
            len0 = None
        if return_robustness:
            return kernel_matrix.cpu(), rhos, selfk, len0
        else:
            return kernel_matrix.cpu()

    def compute_one_bag(self, phi1, phis2, return_robustness=False):
        phis1: list = [phi1]
        return self.compute_bag_bag(phis1, phis2, return_robustness)

    def compute_bag_bag(self, phis1, phis2, return_robustness=False):
        if self.integrate_time:
            rhos1, selfk1, len1 = self._compute_robustness_time(phis1)
            rhos2, selfk2, len2 = self._compute_robustness_time(phis2)
            kernel_matrix = self._compute_kernel_time(
                rhos1, rhos2, selfk1, selfk2, len1, len2
            )
        else:
            rhos1, selfk1 = self._compute_robustness_no_time(phis1)
            rhos2, selfk2 = self._compute_robustness_no_time(phis2)
            len1, len2 = [None, None]
            kernel_matrix = self._compute_kernel_no_time(rhos1, rhos2, selfk1, selfk2)
        if return_robustness:
            return kernel_matrix.cpu(), rhos1, rhos2, selfk1, selfk2, len1, len2
        else:
            return kernel_matrix.cpu()

    def compute_one_from_robustness(self, phi, rhos, rho_self, lengths=None, return_robustness=False):
        phis: list = [phi]
        return self.compute_bag_from_robustness(phis, rhos, rho_self, lengths, return_robustness)

    def compute_bag_from_robustness(self, phis, rhos, rho_self, lengths=None, return_robustness=False):
        if self.integrate_time:
            rhos1, selfk1, len1 = self._compute_robustness_time(phis)
            kernel_matrix = self._compute_kernel_time(
                rhos1, rhos, selfk1, rho_self, len1, lengths
            )
        else:
            rhos1, selfk1 = self._compute_robustness_no_time(phis)
            len1 = None
            kernel_matrix = self._compute_kernel_no_time(rhos1, rhos, selfk1, rho_self)
        if return_robustness:
            return kernel_matrix.cpu(), rhos1, selfk1, len1
        else:
            return kernel_matrix.cpu()

    # def _compute_robustness_time(self, phis):
    #     n = self.samples
    #     p = self.points
    #     k = len(phis)
    #     rhos = torch.zeros((k, n, p), device="cpu")
    #     lengths = torch.zeros(k)
    #     self_kernels = torch.zeros((k, 1))
    #     for i, phi in enumerate(phis):
    #         if self.boolean:
    #             rho = phi.boolean(self.signals, evaluate_at_all_times=True).float()
    #             rho[rho == 0.0] = -1.0
    #         else:
    #             rho = phi.quantitative(self.signals, evaluate_at_all_times=True)
    #         actual_p = rho.size()[2]
    #         rho = rho.reshape(n, actual_p).cpu()
    #         rhos[i, :, :actual_p] = rho
    #         lengths[i] = actual_p
    #         self_kernels[i] = torch.tensordot(
    #             rho.reshape(1, n, -1), rho.reshape(1, n, -1), dims=[[1, 2], [1, 2]]
    #         ) / (actual_p * n)
    #     return rhos, self_kernels, lengths

    def _compute_robustness_no_time(self, phis):
        n = self.samples
        k = len(phis)
        rhos = torch.zeros((k, n))
        self_kernels = torch.zeros((k, 1))

        for i, phi in enumerate(phis):
            if self.boolean:
                rho = phi.eval({
                    "q": self.signals
                }).float().squeeze()
                rho[rho == 0.0] = -1.0
            else:
                rho = phi.robustness({
                    "q": self.signals
                }).squeeze()

            self_kernels[i] = rho.dot(rho) / n
            rhos[i, :] = rho
        return rhos, self_kernels

    def _compute_kernel_time(self, rhos1, rhos2, selfk1, selfk2, len1, len2):
        kernel_matrix = torch.tensordot(rhos1, rhos2, [[1, 2], [1, 2]])
        length_normalizer = self._compute_trajectory_length_normalizer(len1, len2)
        kernel_matrix = kernel_matrix * length_normalizer / self.samples
        if self.normalize:
            kernel_matrix = self._normalize(kernel_matrix, selfk1, selfk2)
        if self.exp_kernel:
            kernel_matrix = self._exponentiate(kernel_matrix, selfk1, selfk2)
        return kernel_matrix

    def _compute_kernel_no_time(self, rhos1, rhos2, selfk1, selfk2):
        kernel_matrix = torch.tensordot(rhos1, rhos2, [[1], [1]])
        kernel_matrix = kernel_matrix / self.samples
        if self.normalize:
            kernel_matrix = self._normalize(kernel_matrix, selfk1, selfk2)
        if self.exp_kernel:
            kernel_matrix = self._exponentiate(kernel_matrix, selfk1, selfk2)
        return kernel_matrix

    @staticmethod
    def _normalize(kernel_matrix, selfk1, selfk2):
        normalize = torch.sqrt(torch.matmul(selfk1, torch.transpose(selfk2, 0, 1)))
        kernel_matrix = kernel_matrix / normalize
        return kernel_matrix

    def _exponentiate(self, kernel_matrix, selfk1, selfk2, sigma2=None):
        if sigma2 is None:
            sigma2 = self.sigma2
        if self.normalize:
            # selfk is (1.0^2 + 1.0^2)
            selfk = 2.0
        else:
            k1 = selfk1.size()[0]
            k2 = selfk2.size()[0]
            selfk = (selfk1 * selfk1).repeat(1, k2) + torch.transpose(
                selfk2 * selfk2, 0, 1
            ).repeat(k1, 1)
        return torch.exp(-(selfk - 2 * kernel_matrix) / (2 * sigma2))

    @staticmethod
    def _compute_trajectory_length_normalizer(len1, len2):
        k1 = len1.size()[0]
        k2 = len2.size()[0]
        y1 = len1.reshape(-1, 1)
        y1 = y1.repeat(1, k2)
        y2 = len2.repeat(k1, 1)
        return 1.0 / torch.min(y1, y2)





class KernelRegression:
    def __init__(
        self,
        kernel,
        cross_validate=False,
        alpha=-2,
        alpha_min=-6,
        alpha_max=1,
        cv_steps=29,
        store_robustness=True,
    ):
        self.kernel = kernel
        self.cross_validate = cross_validate
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.cv_steps = cv_steps
        self.store_robustness = store_robustness
        self.gram = None
        self.train_obs = None
        self.trained = False
        self.weights: Tensor

    def train(
        self,
        train_phis,
        train_obs,
        validate_phis=None,
        validate_obs=None,
        gram=None,
        validate_kernel_vector=None,
    ):
        if gram is None:
            self.gram = GramMatrix(
                self.kernel, train_phis, store_robustness=self.store_robustness
            )
        else:
            self.gram = gram
        self.train_obs = train_obs

        if (
            self.cross_validate
            and validate_phis is not None
            and validate_obs is not None
        ):
            if validate_kernel_vector is None:
                kval = self.gram.compute_bag_kernel_vector(validate_phis)
            else:
                kval = validate_kernel_vector
            cv_par = np.linspace(self.alpha_min, self.alpha_max, self.cv_steps)
            cv_out = np.zeros(self.cv_steps)
            for i, alpha in enumerate(cv_par):
                self._train(alpha)
                pred = torch.matmul(kval, self.weights)
                cv_out[i] = torch.mean((pred - validate_obs) * (pred - validate_obs))
            m = np.argmin(cv_out)
            self.alpha = cv_par[m]
        self._train(self.alpha)
        self.trained = True

    def _train(self, alpha):
        inv = self.gram.invert_regularized(alpha)
        self.weights = torch.matmul(inv, self.train_obs)

    def test(self, test_phis, test_obs, kernel_vector=None):
        if self.trained:
            prediction = self.predict(test_phis, kernel_vector)
            if not prediction.device == test_obs.device:
                prediction.to(test_obs.device)
            mse = torch.mean((prediction - test_obs) * (prediction - test_obs))
            mae = torch.mean(torch.abs(prediction - test_obs))
            # accuracy her means the % of times the sign of prediction is the same as the sign of the true data.
            # accuracy = torch.sum(torch.sign(prediction) == torch.sign(test_obs)).item() /len(test_phis)
            return mse.item(), mae.item(), prediction
        else:
            return None

    def predict(self, phis, kernel_vector=None):
        if self.trained:
            if kernel_vector is None:
                if isinstance(phis, stl.Node):
                    phis: list = [phis]
                kstar = self.gram.compute_bag_kernel_vector(phis)
            else:
                kstar = kernel_vector
            prediction = torch.matmul(kstar, self.weights)
            return prediction
        else:
            return None

    def RKHS_predictor_norm(self):
        if self.trained:
            return torch.matmul(
                self.weights.reshape(1, -1), torch.matmul(self.gram.gram, self.weights)
            )
        else:
            return 0