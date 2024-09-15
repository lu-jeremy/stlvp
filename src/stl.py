"""From Gaia Saveri"""

import numpy as np

from types import SimpleNamespace
from typing import Union

from abc import ABCMeta, abstractmethod
import sys

# For tensor functions
import torch
from torch import Tensor
import torch.nn.functional as F

import amended_stlcg as stlcg

################################################################################
# VARIABLE
################################################################################

class Var:
    def __init__(self, name, dim: int = 1):
        self.name = name
        self.dim = dim

    def get_value(self, env):
        x = env[self.name]

        if x.shape[-1] != self.dim:
            raise ValueError(f'Variable {self.name} expected a value with dimension {self.dim}, but x had shape {x.shape} with dimension {x.shape[-1]}.')

        if x.ndim == 2:
            x = x.unsqueeze(0)

        return x


################################################################################
# STL EXPRESSIONS
################################################################################

class Node(stlcg.DifferentiableSTL, metaclass=ABCMeta):
    """Abstract node class for STL semantics tree."""

    def __init__(self) -> None:
        super().__init__()
        self.props = SimpleNamespace()

    def __getstate__(self):
        state = super().__getstate__()
        # Don't pickle props
        del state["props"]
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        # Add baz back since it doesn't exist in the pickle
        self.props = SimpleNamespace()

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    def boolean(
            self,
            env: dict,
            evaluate_at_all_times: bool = False
    ) -> Tensor:
        """
        Evaluates the boolean semantics at the node.

        Parameters
        ----------
        x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
            The input signals, stored as a batch tensor with three dimensions.
        evaluate_at_all_times: bool
            Whether to evaluate the semantics at all times (True) or
            just at t=0 (False).

        Returns
        -------
        torch.Tensor
        A tensor with the boolean semantics for the node.
        """

        for k, v in env.items():

            if isinstance(v, np.ndarray) and v.ndim == 1:
                v = torch.tensor(v).reshape(-1, 1)
                env[k] = v
            elif isinstance(v, np.ndarray):
                v = torch.tensor(v)
                env[k] = v
            elif isinstance(v, torch.Tensor) and v.ndim == 1:
                v = v.reshape(-1, 1)
                env[k] = v

            if isinstance(v, int):
                v = torch.atleast_2d(torch.tensor(v))
                env[k] = v
            elif isinstance(v, float):
                v = torch.atleast_2d(torch.tensor(v))
                env[k] = v

            if v.ndim > 2:
                raise NotImplementedError()

        z: Tensor = self._boolean(env)
        if evaluate_at_all_times:
            return z
        else:
            return self._extract_semantics_at_time_zero(z)

    def quantitative(
        self,
        env: dict,
        normalize: bool = False,
        evaluate_at_all_times: bool = False,
    ) -> Tensor:
        """
        Evaluates the quantitative semantics at the node.

        Parameters
        ----------
        x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
            The input signals, stored as a batch tensor with three dimensions.
        normalize: bool
            Whether the measure of robustness if normalized (True) or
            not (False). Currently not in use.
        evaluate_at_all_times: bool
            Whether to evaluate the semantics at all times (True) or
            just at t=0 (False).

        Returns
        -------
        torch.Tensor
        A tensor with the quantitative semantics for the node.
        """
        z: Tensor = self._quantitative(env, normalize)
        if evaluate_at_all_times:
            return z
        else:
            return self._extract_semantics_at_time_zero(z)

    def set_normalizing_flag(self, value: bool = True) -> None:
        """
        Setter for the 'normalization of robustness of the formula' flag.
        Currently not in use.
        """
        pass

    def time_depth(self) -> int:
        """Returns time depth of bounded temporal operators only."""
        pass

    @abstractmethod
    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        """Private method equivalent to public one for inner call."""
        raise NotImplementedError

    @abstractmethod
    def _boolean(self, env: dict) -> Tensor:
        """Private method equivalent to public one for inner call."""
        raise NotImplementedError

    @staticmethod
    def _extract_semantics_at_time_zero(x: Tensor) -> Tensor:
        """Extrapolates the vector of truth values at time zero"""
        return x[0]


class Eventually(stlcg.TemporalOperator, Node):
    """Eventually node."""

    def __init__(
        self,
        subformula: Node,
        unbound: bool = False,
        right_unbound: bool = False,
        left_time_bound: int = 0,
        right_time_bound: int = 1,
        adapt_unbound: bool = True,
    ) -> None:
        # TODO: Decide on semantics regarding unbounded time intervals
        # Temporary solution: handle it as a special case in the STL -> LCF code

        super().__init__(
            right_unbound=right_unbound,
            left_time_bound=left_time_bound,
            right_time_bound=right_time_bound+1,
        )
        self.subformula: Node = subformula
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1

        # if self.right_unbound:
        #     self.right_time_bound = np.inf

        self.adapt_unbound: bool = adapt_unbound
        self.subformulas = [subformula]

        self.operation = stlcg.Maxish()

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "eventually" + s0 + " ( " + self.subformula.__str__() + " )"
        return s

    def time_depth(self) -> int:
        if self.unbound:
            return self.subformula.time_depth()
        elif self.right_unbound:
            return self.subformula.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.subformula.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, env: dict) -> Tensor:
        z1: Tensor = self.subformula._boolean(env)
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            z: Tensor = torch.ge(eventually(z1.double(), self.right_time_bound - self.left_time_bound), 0.5)
        return z

    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        z1: Tensor = self.subformula._quantitative(env, normalize)
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            z: Tensor = eventually(z1, self.right_time_bound - self.left_time_bound)
        return z

    # STLCG Temporal Operator Methods

    def _initialize_rnn_cell(self, x):
        '''
        Padding is with the last value of the trace
        '''
        if x.is_cuda:
            self.shift_mat = self.shift_mat.cuda()
            self.append_vec = self.append_vec.cuda()

        h0 = torch.ones([x.shape[0], self.rnn_dim], device=x.device) * x[:, :1]
        count = 0.0

        # if self.interval is [a, np.inf), then the hidden state is a tuple (like in an LSTM)
        if (self._interval[1] == np.inf) & (self._interval[0] > 0):
            d0 = x[:, :1]
            return ((d0, h0.to(x.device)), count)

        return (h0.to(x.device), count)


    def _rnn_cell(self, x, hc, scale=-1, agm=False, distributed=False, **kwargs):
        '''
        x: rnn input [batch_size, 1, ...]
        hc=(h0, c) h0 is the input rnn hidden state  [batch_size, rnn_dim, ...]. c is the count. Initialized by self._initialize_rnn_cell
        '''
        h0, c = hc
        if self.operation is None:
            raise Exception()
        # keeping track of all values that share the min value so the gradients can be distributed equally.

        # Case 1
        if self.interval is None or (self.interval[1] == np.inf) & (self.interval[0] == 0):
            if distributed:
                if x == h0:
                    new_h =  (h0 * c + x) / (c + 1)
                    new_c = c + 1.0
                elif x < h0:
                    new_h = x
                    new_c = 1.0
                else:
                    new_h = h0
                    new_c = c
                state = (new_h, new_c)
                output = new_h
            else:
                input_ = torch.cat([h0, x], dim=1)                                    # [batch_size, rnn_dim+1]
                output = self.operation(input_, scale, dim=1, keepdim=True, agm=agm)  # [batch_size, 1]
                state = (output, None)
        else:
            if (self._interval[1] == np.inf) & (self._interval[0] > 0):
                # Case 3: self.interval is [a, np.inf)
                d0, h0 = h0
                dh = torch.cat([d0, h0[:, :1]], dim=1)                             # [batch_size, 2]
                output = self.operation(dh, scale, dim=1, keepdim=True, agm=agm, distributed=distributed)               # [batch_size, 1, x_dim]

                shifted_h0 = torch.einsum("ij, bj -> bi", self.shift_mat, h0)
                new_elem_vec = (self.append_vec * x).squeeze()
                new_state = shifted_h0 + new_elem_vec
                state = ((output, new_state), None)
            else:
                # Case 2? and 4: self.interval is [a, b]
                shifted_h0 = torch.einsum("ij, bj -> bi", self.shift_mat, h0)
                new_elem_vec = (self.append_vec * x).squeeze()
                state = (shifted_h0 + new_elem_vec, None)

                h0x = torch.cat([h0, x], dim=1)                             # [batch_size, rnn_dim+1]
                input_ = h0x[:, :self.steps]                               # [batch_size, self.steps, x_dim]
                output = self.operation(input_, scale, dim=1, keepdim=True, agm=agm, distributed=distributed)               # [batch_size, 1]
        return output, state


class Always(stlcg.TemporalOperator, Node):
    """Always node."""

    def __init__(
        self,
        subformula: Node,
        unbound: bool = False,
        right_unbound: bool = False,
        left_time_bound: int = 0,
        right_time_bound: int = 1,
        adapt_unbound: bool = True,
    ) -> None:
        super().__init__(
            right_unbound=right_unbound,
            left_time_bound=left_time_bound,
            right_time_bound=right_time_bound+1,
        )
        self.subformula: Node = subformula
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound
        self.subformulas = [subformula]

        self.operation = stlcg.Minish()

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "always" + s0 + " ( " + self.subformula.__str__() + " )"
        return s

    def time_depth(self) -> int:
        if self.unbound:
            return self.subformula.time_depth()
        elif self.right_unbound:
            return self.subformula.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.subformula.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.subformula._boolean(x[:, :, self.left_time_bound:])
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            z: Tensor = torch.ge(eventually(z1.double(), self.right_time_bound - self.left_time_bound), 0.5)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.subformula._quantitative(x[:, :, self.left_time_bound:], normalize)
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            z: Tensor = eventually(z1, self.right_time_bound - self.left_time_bound)
        return z

    # STLCG Temporal Operator Methods

    def _initialize_rnn_cell(self, x):
        '''
        Padding is with the last value of the trace
        '''
        if x.is_cuda:
            self.shift_mat = self.shift_mat.cuda()
            self.append_vec = self.append_vec.cuda()

        h0 = torch.ones([x.shape[0], self.rnn_dim], device=x.device) * x[:, :1]
        count = 0.0
        # if self.interval is [a, np.inf), then the hidden state is a tuple (like in an LSTM)
        if (self._interval[1] == np.inf) & (self._interval[0] > 0):
            d0 = x[:, :1]
            return ((d0, h0.to(x.device)), count)

        return (h0.to(x.device), count)

    def _rnn_cell(self, x, hc, scale=-1, agm=False, distributed=False, **kwargs):
        '''
        x: rnn input [batch_size, 1]
        hc=(h0, c) h0 is the input rnn hidden state  [batch_size, rnn_dim, ...]. c is the count. Initialized by self._initialize_rnn_cell
        '''
        h0, c = hc
        if self.operation is None:
            raise Exception()
        # keeping track of all values that share the min value so the gradients can be distributed equally.
        # TODO: check whether setting this to 1 or np.inf works
        if self.interval is None or (self.interval[1] == 1) & (self.interval[0] == 0):
            if distributed:
                if x == h0:
                    new_h =  (h0 * c + x) / (c + 1)
                    new_c = c + 1.0
                elif x < h0:
                    new_h = x
                    new_c = 1.0
                else:
                    new_h = h0
                    new_c = c
                state = (new_h, new_c)
                output = new_h
            else:
                input_ = torch.cat([h0, x], dim=1)                          # [batch_size, rnn_dim+1]
                output = self.operation(input_, scale, dim=1, keepdim=True, agm=agm)       # [batch_size, 1]
                state = (output, None)
        else: # self.interval is [a, np.inf)
            if (self._interval[1] == np.inf) & (self._interval[0] > 0):
                d0, h0 = h0
                dh = torch.cat([d0, h0[:, :1]], dim=1)                             # [batch_size, 2]
                output = self.operation(dh, scale, dim=1, keepdim=True, agm=agm, distributed=distributed)               # [batch_size, 1, x_dim]

                shifted_h0 = torch.einsum("ij, bj -> bi", self.shift_mat, h0)
                new_elem_vec = (self.append_vec * x).squeeze()
                new_state = shifted_h0 + new_elem_vec
                state = ((output, new_state), None)
            else: # self.interval is [a, b]
                shifted_h0 = torch.einsum("ij, bj -> bi", self.shift_mat, h0)
                new_elem_vec = (self.append_vec * x).squeeze()
                state = (shifted_h0 + new_elem_vec, None)

                h0x = torch.cat([h0, x], dim=1)                             # [batch_size, rnn_dim+1]
                input_ = h0x[:,:self.steps]                               # [batch_size, self.steps, x_dim]
                output = self.operation(input_, scale, dim=1, keepdim=True, agm=agm, distributed=distributed)               # [batch_size, 1]
        return output, state

# TODO:

# class Globally(Node):
#     """Globally class And(Node):
#     """Conjunction node."""

#     def __init__(self, left_child: Node, right_child: Node) -> None:
#         super().__init__()
#         self.left_child: Node = left_child
#         self.right_child: Node = right_child
#         self.subformulas = [left_child, right_child]

#     def __str__(self) -> str:
#         s: str = (
#             "( "
#             + self.left_child.__str__()
#             + " and "
#             + self.right_child.__str__()
#             + " )"
#         )
#         return s

#     def time_depth(self) -> int:
#         return max(self.left_child.time_depth(), self.right_child.time_depth())

#     def _boolean(self, x: Tensor) -> Tensor:
#         z1: Tensor = self.left_child._boolean(x)
#         z2: Tensor = self.right_child._boolean(x)
#         size: int = min(z1.size()[2], z2.size()[2])
#         z1: Tensor = z1[:, :, :size]
#         z2: Tensor = z2[:, :, :size]
#         z: Tensor = torch.logical_and(z1, z2)
#         return z

#     def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
#         z1: Tensor = self.left_child._quantitative(x, normalize)
#         z2: Tensor = self.right_child._quantitative(x, normalize)
#         size: int = min(z1.size()[2], z2.size()[2])
#         z1: Tensor = z1[:, :, :size]
#         z2: Tensor = z2[:, :, :size]
#         z: Tensor = torch.min(z1, z2)
#         return z


class Until(Node):
    """Until node."""

    def __init__(
        self,
        left_child: Node,
        right_child: Node,
        unbound: bool = False,
        right_unbound: bool = False,
        left_time_bound: int = 0,
        right_time_bound: int = 1,
    ) -> None:
        super().__init__()

        self.left_time_bound = left_time_bound

        if right_unbound:
            self.right_time_bound = np.inf
        else:
            self.right_time_bound = right_time_bound

        self._interval = [self.left_time_bound, self.right_time_bound]
        self.interval = self._interval

        assert left_child is not None
        assert right_child is not None

        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.subformulas = [left_child, right_child]

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "( " + self.left_child.__str__() + " until" + s0 + " " + self.right_child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        try:
            sum_children_depth: int = self.left_child.time_depth() + self.right_child.time_depth()
        except AttributeError:
            breakpoint()
            pass

        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return sum_children_depth + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        if self.unbound:
            z1: Tensor = self.left_child._boolean(x)
            z2: Tensor = self.right_child._boolean(x)
            size: int = min(z1.size()[2], z2.size()[2])
            z1: Tensor = z1[:, :, :size]
            z2: Tensor = z2[:, :, :size]
            z1_rep = torch.repeat_interleave(z1.unsqueeze(2), z1.unsqueeze(2).shape[-1], 2)
            z1_tril = torch.tril(z1_rep.transpose(2, 3), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=3)[0]

            z2_rep = torch.repeat_interleave(z2.unsqueeze(2), z2.unsqueeze(2).shape[-1], 2)
            z2_tril = torch.tril(z2_rep.transpose(2, 3), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            z: Tensor = torch.max(torch.min(torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1)[0],
                                  dim=-1)[0]
        elif self.right_unbound:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, right_unbound=True,
                                                   left_time_bound=self.left_time_bound),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._boolean(x)
        else:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, left_time_bound=self.left_time_bound,
                                                   right_time_bound=self.right_time_bound - 1),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._boolean(x)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        if self.unbound:
            z1: Tensor = self.left_child._quantitative(x, normalize)
            z2: Tensor = self.right_child._quantitative(x, normalize)
            size: int = min(z1.size()[2], z2.size()[2])
            z1: Tensor = z1[:, :, :size]
            z2: Tensor = z2[:, :, :size]

            z1_rep = torch.repeat_interleave(z1.unsqueeze(2), z1.unsqueeze(2).shape[-1], 2)
            z1_tril = torch.tril(z1_rep.transpose(2, 3), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=3)[0]

            z2_rep = torch.repeat_interleave(z2.unsqueeze(2), z2.unsqueeze(2).shape[-1], 2)
            z2_tril = torch.tril(z2_rep.transpose(2, 3), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            z: Tensor = torch.max(torch.min(torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1)[0],
                                  dim=-1)[0]
            # z: Tensor = torch.cat([torch.max(torch.min(
            #    torch.cat([torch.cummin(z1[:, :, t:].unsqueeze(-1), dim=2)[0], z2[:, :, t:].unsqueeze(-1)], dim=-1),
            #    dim=-1)[0], dim=2, keepdim=True)[0] for t in range(size)], dim=2)
        elif self.right_unbound:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, right_unbound=True,
                                                   left_time_bound=self.left_time_bound),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._quantitative(x, normalize=normalize)
        else:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, left_time_bound=self.left_time_bound,
                                                   right_time_bound=self.right_time_bound-1),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._quantitative(x, normalize=normalize)
        return z

    def _robustness_trace(self, env, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        '''
        trace1 is the robustness trace of ϕ
        trace2 is the robustness trace of ψ
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        LARGE_NUMBER = 1E6

        interval = self.interval

        trace1 = self.left_child(env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        trace2 = self.right_child(env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)

        always = Always(subformula=self.left_child)

        minish = stlcg.Minish()
        maxish = stlcg.Maxish()

        LHS = trace2.unsqueeze(-1).repeat([1, 1, 1,trace2.shape[1]]).permute(0, 3, 2, 1)
        if interval == None or (self.interval[1] == 1) & (self.interval[0] == 0):
            # Case 1
            RHS = torch.ones_like(LHS)*-LARGE_NUMBER
            for i in range(trace2.shape[1]):
                RHS[:, i:, :, i] = always._run_cell(trace1)

            return maxish(
                minish(torch.stack([LHS, RHS], dim=-1), scale=scale, dim=-1, keepdim=False, agm=agm, distributed=distributed),
                scale=scale, dim=-1, keepdim=False, agm=agm, distributed=distributed
            )

        elif interval[1] < np.Inf:  # [a, b] where b < ∞
            # Case 2
            a = int(interval[0])
            b = int(interval[1])
            RHS = [torch.ones_like(trace1)[:,:b,:] * -LARGE_NUMBER]
            for i in range(b, trace2.shape[1]):
                A = trace2[:, i-b: i-a+1, :].unsqueeze(-1)
                relevant = trace1[:,:i+1,:]
                B = always._run_cell(relevant.flip(1), scale=scale, keepdim=keepdim, distributed=distributed)[:,a:b+1,:].flip(1).unsqueeze(-1)
                RHS.append(maxish(minish(torch.cat([A,B], dim=-1), dim=-1, scale=scale, keepdim=False, distributed=distributed), dim=1, scale=scale, keepdim=keepdim, distributed=distributed))
            return torch.cat(RHS, dim=1)
        else:
            a = int(interval[0])   # [a, ∞] where a < ∞
            RHS = [torch.ones_like(trace1)[:,:a,:] * -LARGE_NUMBER]
            for i in range(a,trace2.shape[1]):
                A = trace2[:,:i-a+1,:].unsqueeze(-1)
                relevant = trace1[:,:i+1,:]
                B = always._run_cell(relevant.flip(1), scale=scale, keepdim=keepdim, distributed=distributed)[:,a:,:].flip(1).unsqueeze(-1)
                RHS.append(maxish(minish(torch.cat([A,B], dim=-1), dim=-1, scale=scale, keepdim=False, distributed=distributed), dim=1, scale=scale, keepdim=keepdim, distributed=distributed))
            return torch.cat(RHS, dim=1)


def eventually(x: Tensor, time_span: int) -> Tensor:
    """
    STL operator 'eventually' in 1D.

    Parameters
    ----------
    x: torch.Tensor
        Signal
    time_span: any numeric type
        Timespan duration

    Returns
    -------
    torch.Tensor
    A tensor containing the result of the operation.
    """
    return F.max_pool1d(x, kernel_size=time_span, stride=1)



class Not(Node):
    """Negation node."""

    def __init__(self, child: Node) -> None:
        super().__init__()
        self.child: Node = child
        self.subformulas = [child]

    def __str__(self) -> str:
        s: str = "not ( " + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        return self.child.time_depth()

    def _boolean(self, env: dict) -> Tensor:
        z: Tensor = ~self.child._boolean(env)
        return z

    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        z: Tensor = -self.child._quantitative(env, normalize)
        return z

    def robustness_trace(self, env, pscale=1, scale=-1, keepdim=True, distributed=False, **kwargs):
        return -self.child(env, pscale=pscale, scale=scale, keepdim=keepdim, distributed=distributed, **kwargs)



class And(Node):
    """Binary Conjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.subformulas = [left_child, right_child]
        self.operation = stlcg.Minish()

    def __str__(self) -> str:
        s: str = (
            "( "
            + self.left_child.__str__()
            + " and "
            + self.right_child.__str__()
            + " )"
        )
        return s

    def time_depth(self) -> int:
        return max(self.left_child.time_depth(), self.right_child.time_depth())

    def _boolean(self, env: dict) -> Tensor:
        z1: Tensor = self.left_child._boolean(env)
        z2: Tensor = self.right_child._boolean(env)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.logical_and(z1, z2)
        return z

    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        z1: Tensor = self.left_child._quantitative(env, normalize)
        z2: Tensor = self.right_child._quantitative(env, normalize)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.min(z1, z2)
        return z

    @staticmethod
    def separate_and(formula, env, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        if formula.__class__.__name__ != "And":
            return formula(env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs).unsqueeze(-1)
        else:
            return torch.cat([
                And.separate_and(formula.left_subformula, env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs),
                And.separate_and(formula.right_subformula, env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
            ], axis=-1)

    def _robustness_trace(self, env, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        xx = torch.cat([
            And.separate_and(self.left_subformula, env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs),
            And.separate_and(self.right_subformula, env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        ], axis=-1)
        return self.operation(xx, scale, dim=-1, keepdim=False, agm=agm, distributed=distributed)                                         # [batch_size, time_dim, ...]



class Or(Node):
    """Binary Disjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_subformula: Node = left_child
        self.right_subformula: Node = right_child
        self.subformulas = [left_child, right_child]
        self.operation = stlcg.Maxish()

    def __str__(self) -> str:
        s: str = (
            "( "
            + self.left_subformula.__str__()
            + " or "
            + self.right_subformula.__str__()
            + " )"
        )
        return s

    def time_depth(self) -> int:
        return max(self.left_subformula.time_depth(), self.right_subformula.time_depth())

    def _boolean(self, env: dict) -> Tensor:
        z1: Tensor = self.left_subformula._boolean(env)
        z2: Tensor = self.right_subformula._boolean(env)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.logical_or(z1, z2)
        return z

    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        z1: Tensor = self.left_subformula._quantitative(env, normalize)
        z2: Tensor = self.right_subformula._quantitative(env, normalize)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.max(z1, z2)
        return z

    @staticmethod
    def separate_or(formula, env, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        if formula.__class__.__name__ != "And":
            return formula(env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs).unsqueeze(-1)
        else:
            return torch.cat([
                Or.separate_or(formula.left_subformula, env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs),
                Or.separate_or(formula.right_subformula, env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
            ], axis=-1)

    def _robustness_trace(self, env, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        xx = torch.cat([
            Or.separate_or(self.left_subformula, env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs),
            Or.separate_or(self.right_subformula, env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        ], axis=-1)
        return self.operation(xx, scale, dim=-1, keepdim=False, agm=agm, distributed=distributed)                                         # [batch_size, time_dim, ...]


class Atom(Node):
    """A leaf node of an STL expression, which can actually transform and
    evaluate the signal at every sampled time to produce a boolean result (or a
    robustness measurement).

    """

    def __init__(self) -> None:
        super().__init__()
        self.subformulas = []

    def time_depth(self) -> int:
        return 0


class DistComparison(Atom):

    def __init__(self, var: Var, landmark, threshold):
        super().__init__()
        self.var = var
        self.landmark = landmark
        self.threshold = threshold

    def __str__(self):
        pass

    def _boolean(self):
        pass

    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        pass

    def _robustness_trace(self, env: dict, normalize=False, pscale=1.0, **kwargs):
        traj = self.var.get_value(env)

        # l2norm
        distance = (traj - self.landmark).norm(dim=-1)

        # quantitative robustness for comparison
        z = (-distance + self.threshold)

        if normalize:
            z: Tensor = torch.tanh(z)

        return z




class Comparison(Atom):
    """One kind of atomic formula of the form X<=t or X>=t. X is not necessarily
    one dimensional"""

    def __init__(self, var: Var, threshold, lte: bool = True) -> None:
        super().__init__()
        self.var = var
        self.threshold = threshold
        self.lte: bool = lte

    def __str__(self) -> str:
        s: str = (
            self.var.name
            + (" <= " if self.lte else " >= ")
            + str(self.threshold)
        )
        return s

    def _boolean(self, env: dict) -> Tensor:
        x = self.var.get_value(env) # horizon x dim

        if self.lte:
            z = (x <= self.threshold).all(dim=-1)
        else:
            z = (x >= self.threshold).all(dim=-1)

        return z

    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        x = self.var.get_value(env) # horizon x dim

        if self.lte:
            z = (-x + self.threshold).min(dim=-1)
        else:
            z = (x - self.threshold).min(dim=-1)

        if normalize:
            z: Tensor = torch.tanh(z)

        return z

    def _robustness_trace(self, env: dict, pscale=1.0, **kwargs):
        '''
        Computing robustness trace.
        pscale scales the robustness by a constant. Default pscale=1.
        '''
        return self._quantitative(env)

def multiply_matrix_over_trajectory_batch(matrix, trajs):
    return torch.einsum("ij, bhj -> bhi", matrix, trajs)


class LinearExp(Atom):
    """Linear expression that states A @ x <= b, This can be used to test
    whether the point x is in a polytope. In other words, x is in Poly(H, b) if
    b - H @ x > 0, in which case A = H, and b = b.
    """

    def __init__(self, var: Var, A, b) -> None:
        super().__init__()
        self.var = var
        self.register_buffer('A', A)
        self.register_buffer('b', b)

    def __str__(self) -> str:
        return f"A @ {self.var.name} <= b"

    def _boolean(self, env: dict) -> Tensor:
        x = self.var.get_value(env) # batch x horizon x dim
        prods_over_horizon = multiply_matrix_over_trajectory_batch(self.A, x)
        x = prods_over_horizon - self.b # horizon x dim2 - (1, dim2)
        z = (x <= 0).all(dim=-1) # horizon
        return z

    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        x = self.var.get_value(env) # batch x horizon x dim
        prods_over_horizon = multiply_matrix_over_trajectory_batch(self.A, x)
        x = prods_over_horizon - self.b # horizon x dim2 - (1, dim2)
        z = (-x).min(dim=-1).values # horizon

        if normalize:
            z: Tensor = torch.tanh(z)

        return z

    def _robustness_trace(
            self,
            env: dict,
            pscale: float = 1.0,
            **kwargs
    ):
        return self._quantitative(env) * pscale


class NegLinearExp(Atom):
    """Linear expression that states any of the rows from (A @ x) are >= b, This
    can be used to test whether the point x is outside a polytope.
    """

    def __init__(self, var: Var, A, b) -> None:
        super().__init__()
        self.var = var
        self.register_buffer('A', A)
        self.register_buffer('b', b)

    def __str__(self) -> str:
        return f"A @ {self.var.name} >= b (or)"

    def _boolean(self, env: dict) -> Tensor:
        x = self.var.get_value(env) # horizon x dim
        prods_over_horizon = multiply_matrix_over_trajectory_batch(self.A, x)
        x = prods_over_horizon - self.b # horizon x dim2 - (1, dim2)
        z = (x >= 0).any(dim=-1) # horizon
        return z

    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        x = self.var.get_value(env) # batches x horizon x dim
        prods_over_horizon = multiply_matrix_over_trajectory_batch(self.A, x)
        x = prods_over_horizon - self.b # horizon x dim2 - (1, dim2)
        z = x.max(dim=-1).values # horizon

        if normalize:
            z: Tensor = torch.tanh(z)

        return z

    def _robustness_trace(
            self,
            env: dict,
            pscale: float = 1.0,
            **kwargs
    ):
        return self._quantitative(env) * pscale