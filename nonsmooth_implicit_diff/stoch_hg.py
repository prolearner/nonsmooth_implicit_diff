import torch
from torch.autograd import grad as torch_grad
from torch import Tensor
from typing import Union, List, Callable


def NSID(
    params: List[Tensor],
    hparams: List[Tensor],
    step_size: Union[float, Callable[[int], float]],
    K: int,
    T_hat_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
    T_bar_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
    G_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
    outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
    tol=1e-10,
    set_grad=True,
    verbose=True,
    print_interval=10,
) -> List[Tensor]:
    
    if isinstance(step_size, float):
        get_stepsize = lambda x: step_size
    else:
        get_stepsize = step_size


    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    w_bar = T_bar_map(params, hparams)
    w_bar_G_hparams = G_map(w_bar, hparams)

    w_bar_detached = [w.detach().requires_grad_(True) for w in w_bar]
    w_bar_G_params = G_map(w_bar_detached, hparams)

    def psi(vs):
        w_up = T_hat_map(params, hparams)
        Jvp_G_bar_vs = torch_grad(
            w_bar_G_params, w_bar_detached, grad_outputs=vs, retain_graph=True
        )
        Jvp_T_vs = torch_grad(w_up, params, grad_outputs=Jvp_G_bar_vs, retain_graph=False)
        return [v + gow for v, gow in zip(Jvp_T_vs, grad_outer_w)]
    

    vs = [torch.zeros_like(w) for w in params]
    vs_vec = cat_list_to_tensor(vs)
    for k in range(K):
        vs_prev_vec = vs_vec

        lr = get_stepsize(k)
        vs = [(1 - lr) * v + lr * psi_v for v, psi_v in zip(vs, psi(vs))]

        vs_vec = cat_list_to_tensor(vs)
        norm_diff = torch.norm(vs_vec - vs_prev_vec)
        if verbose and (k %print_interval == 0 or k==K-1):
            print(f"ls iter {k}: ||vs - vs_prec|| = {norm_diff:.4e}")
            
        
        if float(norm_diff) < tol:
            break
            
    grads = torch_grad(w_bar_G_hparams, hparams, grad_outputs=vs, allow_unused=True)
    grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads


# UTILS


def grd(a, b):
    return torch.autograd.grad(a, b, create_graph=True, retain_graph=True)


def list_dot(l1, l2):  # extended dot product for lists
    return torch.stack([(a * b).sum() for a, b in zip(l1, l2)]).sum()


def jvp(fp_map, params, vs):
    dummy = [torch.ones_like(phw).requires_grad_(True) for phw in fp_map(params)]
    g1 = grd(list_dot(fp_map(params), dummy), params)
    return grd(list_dot(vs, g1), dummy)


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(
        outer_loss, hparams, retain_graph=retain_graph
    )

    return grad_outer_w, grad_outer_hparams


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g


def grad_unused_zero(
    output, inputs, grad_outputs=None, retain_graph=False, create_graph=False
):
    grads = torch.autograd.grad(
        output,
        inputs,
        grad_outputs=grad_outputs,
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))
