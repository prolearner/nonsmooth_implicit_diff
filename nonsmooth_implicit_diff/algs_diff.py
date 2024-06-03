from collections import defaultdict
import math
import torch
import numpy as np
from nonsmooth_implicit_diff.data_loader import FastTensorDataLoader
from torch.nn import functional as F
from torch.func import grad

from nonsmooth_implicit_diff.utils import copy_tensor_list, vectorize


def accuracy(y_pred, y):
    return (torch.argmax(y_pred, dim=1) == y).float().mean().item()


def square_loss(x, y, w):
    return 0.5 * torch.mean((x @ w - y) ** 2)


def square_loss_gradient(x, y, w):
    return (1 / len(y)) * x.T @ (x @ w - y)


def cross_entropy(x, y, w):
    return F.cross_entropy(x @ w, y)


def soft_thresholding(x, gamma):
    if gamma == 0:
        return x

    return torch.sign(x - gamma) * torch.maximum(
        torch.zeros_like(x), torch.abs(x) - gamma
    )


def compute_L_mu(x):
    eigenvalues, _ = np.linalg.eigh(x.T @ x / x.shape[0])
    L = np.max(eigenvalues)
    mu = np.maximum(0, np.min(eigenvalues))
    return L, mu


def compute_opt_step_size(alpha_l2, L=None, mu=None):
    # alpha_l2 is the l2-regularization parameter
    return 2 / (L + mu + 2 * alpha_l2)


class Algorithm:
    def __init__(self, x, y, x_val, y_val, compute_eigenvalues=True, device="cpu"):
        self.x = x.to(device)
        self.y = y.to(device)
        self.x_val = x_val.to(device)
        self.y_val = y_val.to(device)
        self.eigenvalues = None
        self.batch_size = self.x.shape[0]
        if compute_eigenvalues:
            self.L, self.mu = compute_L_mu(x)
            self.cond = self.L / self.mu if self.mu > 0 else np.inf
            print(f"Problem with {self.L=}, {self.mu=}, {self.cond=}")
        self.device = device

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, x, y, w):
        raise NotImplementedError

    def train_loss(self, params, hparams):
        raise NotImplementedError

    def train_acc(self, params):
        return accuracy(self.x @ params[0], self.y)

    def val_loss(self, params, hparams):
        raise NotImplementedError

    def val_acc(self, params):
        return accuracy(self.x_val @ params[0], self.y_val)

    def phi(self, params, hparams):
        raise NotImplementedError

    def G_map(self, params, hparams):
        return params

    def params_init(self):
        raise NotImplementedError

    def run(
        self,
        hparams,
        n_iter=1,
        print_interval=1,
        step_size=1,
        callback=None,
        classification=False,
        check_contraction=False,
    ):
        if not callable(step_size):
            step_size_f = lambda x: step_size
        else:
            step_size_f = step_size

        results = defaultdict(list)

        def eval(params, t):
            train_loss = self.train_loss(params, hparams)
            val_loss = self.val_loss(params, hparams)
            results["history"].append(copy_tensor_list(params))
            results["train_loss"].append(float(train_loss.detach().cpu().numpy()))
            results["val_loss"].append(float(val_loss.detach().cpu().numpy()))
            if classification:
                results["train_acc"].append(float(self.train_acc(params)))
                results["val_acc"].append(float(self.val_acc(params)))
            results["t"].append(t)
            results["n_samples"].append(t * self.batch_size)
            if callback is not None:
                callback(params, t, results)

        def print_losses(results, i):
            train_loss = results["train_loss"][-1]
            val_loss = results["val_loss"][-1]
            if classification:
                train_acc = results["train_acc"][-1]
                val_acc = results["val_acc"][-1]
                print(f"t={i}: {train_loss=}, {val_loss=} | {train_acc=}, {val_acc=}")
            else:
                print(f"t={i}: {train_loss=}, {val_loss=}")

        params = [p.to(self.device) for p in self.params_init()]
        d = len(vectorize(params))
        eval(params, 0)
        print_losses(results, 0)
        norm_prec = float("inf")
        for i in range(n_iter):
            step_size = step_size_f(i)
            if step_size == 1:
                params = self.G_map(self.phi(params, hparams), hparams)
            else:
                params = [
                    (1 - step_size) * p + step_size * z
                    for p, z in zip(params, self.phi(params, hparams))
                ]
                params = self.G_map(params, hparams, eta=step_size)
            eval(params, i + 1)

            if print_interval is not None and (
                i % print_interval == 0 or i == n_iter - 1
            ):
                print_losses(results, i + 1)

            norm_diff = torch.norm(
                vectorize(results["history"][-1]) - vectorize(results["history"][-2])
            )
            if check_contraction and norm_diff > 1e-7 * math.sqrt(d) + norm_prec:
                raise ValueError(f"At iteration {i} the contraction was not satisfied")

            norm_prec = norm_diff

        return results


class ISTA(Algorithm):
    def __init__(
        self, x, y, x_val, y_val, step_size="optimal", loss="square", device="cpu"
    ):
        super().__init__(x, y, x_val, y_val, device=device)
        self.mode = loss
        if loss == "square":
            self.loss = square_loss
            self.grad_loss = square_loss_gradient
            self.params_shape = (x.shape[-1],)
        elif loss == "cross_entropy":
            self.loss = cross_entropy
            self.grad_loss = grad(cross_entropy, argnums=2)
            self.mu = self.mu
            self.params_shape = (x.shape[1], len(torch.unique(y)))
        else:
            raise NotImplementedError

        self.step_size = None
        self.reset(step_size)

    def params_init(self):
        return [torch.zeros(*self.params_shape)]

    def run(
        self,
        hparams,
        n_iter=1,
        print_interval=1,
        step_size=1,
        callback=None,
        check_contraction=False,
    ):
        def new_callback(params, t, results):
            results["sparsity_frac"].append(
                float((params[0] == 0).float().mean().detach().cpu().numpy())
            )
            if callable(callback):
                callback(params, t, results)

        classification = True if self.mode == "cross_entropy" else False
        return super().run(
            hparams,
            n_iter,
            print_interval,
            step_size,
            new_callback,
            classification=classification,
            check_contraction=check_contraction,
        )

    def reset(self, step_size=None):
        if step_size == "optimal":

            def step_size(hparams, t=None):
                return compute_opt_step_size(alpha_l2=hparams[1], L=self.L, mu=self.mu)

        if step_size is not None:
            self.step_size = step_size
        else:
            raise ValueError

    def train_loss(self, params, hparams):
        w = params[0]
        alpha_l1, alpha_l2 = hparams
        return (
            self.loss(self.x, self.y, w)
            + 0.5 * alpha_l2 * torch.sum(w**2)
            + alpha_l1 * torch.sum(w.abs())
        )

    def val_loss(self, params, hparams):
        return self.loss(self.x_val, self.y_val, params[0])

    def gd_step(self, w, alpha_l2, step_size):
        return w - step_size * (self.grad_loss(self.x, self.y, w) + alpha_l2 * w)

    def phi(self, params, hparams):
        w = params[0]
        alpha_l1, alpha_l2 = hparams

        if callable(self.step_size):
            step_size = self.step_size(hparams)
        else:
            step_size = self.step_size

        w_up = self.gd_step(w, alpha_l2, step_size)
        return [soft_thresholding(w_up, alpha_l1 * step_size)]


class ISTA_stoch(ISTA):
    def __init__(
        self,
        x,
        y,
        x_val,
        y_val,
        step_size,
        batch_size,
        shuffle=True,
        loss="square",
        device="cpu",
    ):
        self.loader = FastTensorDataLoader(
            x, y, batch_size=batch_size, shuffle=shuffle, device=device
        )
        super().__init__(x, y, x_val, y_val, step_size, loss=loss, device=device)

        self.batch_size = batch_size
        self.iter_counter = None
        self.epoch_counter = None
        self.current_batch = None
        self.reset(step_size)

    def reset(self, step_size=None):
        self.iter_counter = 0
        self.epoch_counter = 0
        self.loader = iter(self.loader)
        super().reset(step_size)

    def train_loss_current_batch(self, params, hparams):
        w = params[0]
        alpha_l1, alpha_l2 = hparams
        return (
            self.loss(*self.current_batch, w)
            + 0.5 * alpha_l2 * torch.sum(w**2)
            + alpha_l1 * torch.sum(w.abs())
        )

    def gd_step(self, w, alpha_l2, step_size):
        return w - step_size * (self.grad_loss(*self.current_batch, w) + alpha_l2 * w)

    def G_map(self, params, hparams, eta=1):
        alpha_l1, _ = hparams
        if callable(self.step_size):
            step_size = self.step_size(hparams)
        else:
            step_size = self.step_size

        return [soft_thresholding(params[0], alpha_l1 * step_size * eta)]

    def phi_bar_map(self, params, hparams, J):
        ts = [self.phi(params, hparams)[0] for _ in range(J)]
        return [torch.mean(torch.stack(ts), dim=0)]

    def phi(self, params, hparams):
        try:
            self.current_batch = next(self.loader)
        except StopIteration:
            self.loader = iter(self.loader)
            self.current_batch = next(self.loader)
            self.epoch_counter += 1
        self.iter_counter += 1

        w = params[0]
        alpha_l1, alpha_l2 = hparams
        if callable(self.step_size):
            step_size = self.step_size(hparams)
        else:
            step_size = self.step_size

        return [self.gd_step(w, alpha_l2, step_size)]


class ISTA_poison(Algorithm):
    def __init__(
        self,
        x,
        y,
        x_val,
        y_val,
        poisoned_indices,
        step_size="optimal",
        loss="square",
        device="cpu",
        alpha_l1=0.1,
        alpha_l2=0.1,
    ):
        super().__init__(x, y, x_val, y_val, device=device)
        assert max(poisoned_indices) < x.shape[0]

        self.poisoned_indices = poisoned_indices
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2

        self.mode = loss
        if loss == "square":
            self.loss = square_loss
            self.grad_loss = square_loss_gradient
            self.params_shape = (x.shape[-1],)
        elif loss == "cross_entropy":
            self.loss = cross_entropy
            self.grad_loss = grad(cross_entropy, argnums=2)
            self.mu = self.mu
            self.params_shape = (x.shape[1], len(torch.unique(y)))
        else:
            raise NotImplementedError

        self.step_size = None
        self.reset(step_size)

    def poison_dataset(self, poison, indices=None):
        x_poisoned = self.x + torch.zeros_like(
            self.x
        )  # Create a new tensor with the same shape as x
        x_poisoned[self.poisoned_indices] += poison
        return x_poisoned, self.y

    def params_init(self):
        return [torch.zeros(*self.params_shape)]

    def run(
        self,
        hparams,
        n_iter=1,
        print_interval=1,
        step_size=1,
        callback=None,
        check_contraction=False,
    ):
        def new_callback(params, t, results):
            results["sparsity_frac"].append(
                float((params[0] == 0).float().mean().detach().cpu().numpy())
            )
            if callable(callback):
                callback(params, t, results)

        classification = True if self.mode == "cross_entropy" else False
        return super().run(
            hparams,
            n_iter,
            print_interval,
            step_size,
            new_callback,
            classification=classification,
            check_contraction=check_contraction,
        )

    def reset(self, step_size=None):
        if step_size == "optimal":

            def step_size(hparams, t=None):
                return compute_opt_step_size(
                    alpha_l2=self.alpha_l2, L=self.L, mu=self.mu
                )

        if step_size is not None:
            self.step_size = step_size
        else:
            raise ValueError

    def train_loss(self, params, hparams):
        x_poisoned, y = self.poison_dataset(hparams[0])
        w = params[0]
        alpha_l1, alpha_l2 = self.alpha_l1, self.alpha_l2
        return (
            self.loss(x_poisoned, y, w)
            + 0.5 * alpha_l2 * torch.sum(w**2)
            + alpha_l1 * torch.sum(w.abs())
        )

    def val_loss(self, params, hparams):
        return -self.loss(self.x_val, self.y_val, params[0])

    def gd_step(self, w, alpha_l2, step_size, x, y):
        return w - step_size * (self.grad_loss(x, y, w) + alpha_l2 * w)

    def phi(self, params, hparams):
        x_poisoned, y = self.poison_dataset(hparams[0])
        w = params[0]
        alpha_l1, alpha_l2 = self.alpha_l1, self.alpha_l2

        if callable(self.step_size):
            step_size = self.step_size(hparams)
        else:
            step_size = self.step_size

        w_up = self.gd_step(w, alpha_l2, step_size, x_poisoned, y=y)
        return [soft_thresholding(w_up, alpha_l1 * step_size)]


class ISTA_poison_stoch(ISTA_poison):
    def __init__(
        self,
        x,
        y,
        x_val,
        y_val,
        batch_size,
        poisoned_indices,
        step_size="optimal",
        alpha_l1=0.1,
        alpha_l2=0.1,
        shuffle=True,
        loss="square",
        device="cpu",
    ):
        self.loader = FastTensorDataLoader(
            torch.arange(x.shape[0]), batch_size=batch_size, shuffle=shuffle, device=device
        )
        super().__init__(x, y, x_val, y_val, step_size=step_size, loss=loss, device=device,
                         poisoned_indices=poisoned_indices,
                         alpha_l1=alpha_l1, alpha_l2=alpha_l2)

        self.batch_size = batch_size
        self.iter_counter = None
        self.epoch_counter = None
        self.current_batch = None
        self.reset(step_size)
    
    def poison_dataset(self, poison, indices=None):
        if indices is None:
            return super().poison_dataset(poison)
        poison_mask = torch.zeros_like(self.x) 
        poison_mask[self.poisoned_indices] += poison
        x_poisoned_batch = self.x[indices] + poison_mask[indices]
        return x_poisoned_batch, self.y[indices]
        

    def reset(self, step_size=None):
        self.iter_counter = 0
        self.epoch_counter = 0
        self.loader = iter(self.loader)
        super().reset(step_size)

    def train_loss_current_batch(self, params, hparams):
        w = params[0]
        alpha_l1, alpha_l2 = self.alpha_l1, self.alpha_l2
        return (
            self.loss(*self.current_batch, w)
            + 0.5 * alpha_l2 * torch.sum(w**2)
            + alpha_l1 * torch.sum(w.abs())
        )

    def G_map(self, params, hparams, eta=1):
        alpha_l1 = self.alpha_l1
        if callable(self.step_size):
            step_size = self.step_size(hparams)
        else:
            step_size = self.step_size

        return [soft_thresholding(params[0], alpha_l1 * step_size * eta)]

    def phi_bar_map(self, params, hparams, J):
        ts = [self.phi(params, hparams)[0] for _ in range(J)]
        return [torch.mean(torch.stack(ts), dim=0)]

    def phi(self, params, hparams):
        try:
            current_batch_idx = next(self.loader)
        except StopIteration:
            self.loader = iter(self.loader)
            current_batch_idx = next(self.loader)
            self.epoch_counter += 1
        self.iter_counter += 1
        
        self.current_batch = self.poison_dataset(hparams[0], indices=current_batch_idx)

        w = params[0]
        alpha_l1, alpha_l2 = self.alpha_l1, self.alpha_l2
        if callable(self.step_size):
            step_size = self.step_size(hparams)
        else:
            step_size = self.step_size

        return [self.gd_step(w, alpha_l2, step_size, *self.current_batch)]
