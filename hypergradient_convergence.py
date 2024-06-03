import copy
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import hypergrad as hg
from collections import defaultdict
from progressbar import progressbar
from os.path import join
from nonsmooth_implicit_diff.algs_diff import (
    ISTA,
    ISTA_poison,
    ISTA_poison_stoch,
    ISTA_stoch,
    compute_L_mu,
)
from nonsmooth_implicit_diff.data_generation import make_sparse_regression, load_minst
from nonsmooth_implicit_diff.plot_utils import (
    plot_hg_convergence,
    plot_problem_features,
    plot_ll_convergence,
)
from nonsmooth_implicit_diff.utils import (
    copy_tensor_list,
    make_exp_folder,
    set_seed,
    to_list,
    to_np,
    to_torch,
    vectorize,
    when_is_support_identified,
)
from nonsmooth_implicit_diff.stoch_hg import NSID



def main():
    run_elastic_net(save_plots=True)


def run_elastic_net(
    mode="last_det",  # OPTIONS:  normal last last_det
    data="mnist",  # OPTIONS: mnist, synth
    problem_type="HPO", # OPTIONS: poisoning, HPO
    poisoned_frac=.3,
    random_state=42,
    n_samples=500,
    n_features=100,
    noise=0.1,
    effective_rank=30,
    n_informative=30,
    correlated=False,
    shuffle=False,
    val_size=0.4,
    test_size=0.4,
    alpha_l1=0.1,
    alpha_l2=0.1,
    max_iter_ll=2000,
    max_iter_ll_stoch=4000,
    max_iter_hg=2000,
    max_iter_hg_stoch=4000,
    step_size_phi="optimal",
    opt_L_mu_div=1.0,
    J_inner_mult = 1.,
    beta_ll=1,
    gamma_ll=100,
    beta_mult=1.,
    gamma_mult=1.,
    batch_size_ll=100,
    eval_inteval_ll=100,
    n_eval_points=10,
    max_iter_true=10000,
    show_plots=False,
    save_plots=False,
    save_path='tests',
    device="cpu",
    compute_stochastic_grads=False,
):
    assert not (save_plots and save_path is None)
    
    shuffle_optim = True

    conf = locals().copy()

    set_seed(random_state)

    synth_params = (
        "n_features",
        "n_informative",
        "noise",
    )

    params_in_name = (
        "problem_type",
        "data",
        "mode",
        "alpha_l1",
        "alpha_l2",
        "n_samples",
        "random_state",
    )
    
    if data == "synth":
        params_in_name = (*params_in_name, *synth_params)
    else:
        conf = {k: v for k, v in conf.items() if k not in synth_params}

    print(conf)

    if data == "synth":
        X_train, X_val, X_test, y_train, y_val, y_test, w_true = make_sparse_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            effective_rank=effective_rank,
            n_informative=n_informative,
            correlated=correlated,
            shuffle=shuffle,
            random_state=random_state,
            test_size=test_size,
            val_size=val_size,
        )
        loss = "square"
    else:
        X_train, X_val, X_test, y_train, y_val, y_test, w_true = load_minst(
            n_samples=n_samples,
            shuffle=shuffle,
            random_state=random_state,
            val_size=val_size,
        )
        loss = "cross_entropy"


    common_algo_params = dict(
        x=X_train, y=y_train, x_val=X_val, y_val=y_val, loss=loss, device=device
    )
    
    ALGO_DET_CLASS = ISTA
    ALGO_STOCH_CLASS = ISTA_stoch
    if problem_type == "poisoning":
        poisoned_indices = torch.randint(0, X_train.shape[0] , (int(X_train.shape[0]*poisoned_frac),), ).to(device)
        hparams = [torch.rand_like(X_train[poisoned_indices.cpu()]).to(device)]
        hparams[0] = torch.clamp(hparams[0],-.1, .1)

        ALGO_DET_CLASS = ISTA_poison
        ALGO_STOCH_CLASS = ISTA_poison_stoch
        common_algo_params = dict(
            **common_algo_params,
            alpha_l1 =alpha_l1, alpha_l2=alpha_l2,
            poisoned_indices=poisoned_indices,
        )

    elif problem_type =="HPO":
        hparams = [torch.tensor(alpha_l1, dtype=torch.float32), torch.tensor(alpha_l2, dtype=torch.float32)]
        
    
    ll_algos = dict()

    ll_algos["opt"] = ALGO_DET_CLASS(**common_algo_params, step_size="optimal"), dict(
        n_iter=max_iter_true,
        print_interval=eval_inteval_ll,
        # check_contraction=True,
    )
    
    if problem_type == "poisoning":
        X_poison = ll_algos["opt"][0].poison_dataset(hparams[0])[0]
        L_min, mu_min = compute_L_mu(X_poison.cpu().detach().numpy())
        L_min, mu_min = float(L_min/opt_L_mu_div), float(mu_min/opt_L_mu_div)
        def step_size_opt(hparams):
            return 2/(L_min+mu_min + 2*alpha_l2)

    elif problem_type=="HPO":
        L_min, mu_min = ll_algos["opt"][0].L/opt_L_mu_div, ll_algos["opt"][0].mu/opt_L_mu_div
        def step_size_opt(hparams):
            alpha_l2 = hparams[1]
            return 2/(L_min+mu_min + 2*alpha_l2)
    else:
        raise NotImplementedError
    
    ll_algos['opt'][0].step_size = step_size_opt
    step_size_opt_value = step_size_opt([alpha_l1, alpha_l2])

    q = max(abs(1-step_size_opt_value*(L_min + alpha_l2)), abs(1-step_size_opt_value*(mu_min + alpha_l2)))
    
    beta_opt = 2 / (1 - q**2)
    gamma_opt = beta_opt

    if beta_ll == "optimal":
        beta_ll = beta_opt*beta_mult
        gamma_ll = gamma_opt*gamma_mult

    print(f"Using {beta_ll=}, {gamma_ll=}")

    step_size_stoch_dec = lambda x: beta_ll / (gamma_ll + x)
    step_size_ll = step_size_stoch_dec(0)

    if compute_stochastic_grads:
        ll_algos["stoch_const"] = ALGO_STOCH_CLASS(
            **common_algo_params,
            step_size=step_size_phi,
            batch_size=batch_size_ll,
            shuffle=shuffle_optim,
        ), dict(
            n_iter=max_iter_ll_stoch,
            print_interval=eval_inteval_ll,
            step_size=step_size_ll,
        )

        ll_algos["stoch_dec"] = ll_algos["stoch_const"][0], dict(
            n_iter=max_iter_ll_stoch,
            print_interval=eval_inteval_ll,
            step_size=step_size_stoch_dec,
        )

    if not step_size_phi == "optimal":
        ll_algos["det"] = ALGO_DET_CLASS(**common_algo_params, step_size=step_size_phi), dict(
            n_iter=max_iter_ll,
            print_interval=eval_inteval_ll,
        )


    # All parameters have been set, conf can be updated and saved to the experiment directory
    conf["step_size_opt"] = float(step_size_opt_value)
    conf["true_beta_ll"] = float(beta_ll)
    conf["true_gamma_ll"] = float(gamma_ll)
    conf["const_step_size_ll"] = float(step_size_ll)
    conf["lambda_max_min_hess_loss"] = (
        float(ll_algos["opt"][0].L),
        float(ll_algos["opt"][0].mu),
    )
    conf["L_mu_q"] = (float(L_min), float(mu_min), float(q))

    if save_path is not None:
        save_path = make_exp_folder(conf, params_in_name, save_path)

    if show_plots or save_plots:
        plot_problem_features(
            X_train,
            y_train,
            w_true,
            save_plot=save_plots,
            show_plot=show_plots,
            save_path=save_path,
        )


    ll_results = dict()
    for algo_name, (algo, algo_kwparams) in ll_algos.items():
        print(f"Running algo {algo_name}")
        ll_results[algo_name] = algo.run(
            hparams=hparams,
            **algo_kwparams,
        )

    if "det" not in ll_results:
        ll_results["det"] = {k: v[:max_iter_ll] for k, v in ll_results["opt"].items()}
        ll_algos["det"] = ll_algos["opt"]

    params_true = ll_results["opt"]["history"][-1]
    ll_results = {k: v for k, v in ll_results.items() if k != "opt"}

    ll_algo_opt = ll_algos["opt"][0]
    history_det = ll_results["det"]["history"]

    if compute_stochastic_grads:
        ll_algo_stoch = ll_algos["stoch_const"][0]
        history_stoch_const = ll_results["stoch_const"]["history"]
        history_stoch_dec = ll_results["stoch_dec"]["history"]

    for algo_name, res in ll_results.items():
        print(f"Computing results for {algo_name}")
        res["diff_w"] = [
            torch.norm(p[0] - params_true[0]).cpu() for p in res["history"]
        ]
        res["diff_w_norm"] = [
            (torch.norm(p[0] - params_true[0]) / torch.norm(params_true[0])).cpu()
            for p in res["history"]
        ]

        (
            support_identification_index,
            matching_vector,
            lost_support,
        ) = when_is_support_identified(
            w_true=params_true[0].cpu().numpy(),
            params_history=[p[0].cpu().numpy() for p in res["history"]],
        )

        if support_identification_index is not None:
            print(
                f"  Support was identified after {support_identification_index}/{max_iter_ll+1} iterations"
            )
        if lost_support is True:
            print("Support was lost after some iterations")

        res["support_identification_index"] = support_identification_index
        res["matching_vector"] = matching_vector

    if show_plots or save_plots:
        plot_ll_convergence(
            ll_results,
            params_true,
            save_plot=save_plots,
            show_plot=show_plots,
            save_path=save_path,
        )

    for algo_name in ll_results:
        ll_results[algo_name] = {
            k: v for k, v in ll_results[algo_name].items() if k != "history"
        }

    hparams = [tensor.requires_grad_(True) for tensor in copy_tensor_list(hparams)]

    print("Computing true hypergradient...")
    true_hypergrad = vectorize(
        hg.fixed_point(
            params_true,
            hparams,
            max_iter_true,
            ll_algo_opt.phi,
            ll_algo_opt.val_loss,
            # check_contraction=True,
            # verbose=True,
        )
    )

    def get_wt_and_phi(t, mode, method):
        # get the last iterates and phi maps for deterministic_gradients
        if mode == "normal":
            if "reverse" in method:
                return history_det[: t + 1], [ll_algos["det"][0].phi] * t
            else:
                return history_det[t], ll_algos["det"][0].phi
        if mode == "last_det":
            if "reverse" in method:
                return history_det[-t - 1 :], [ll_algos["det"][0].phi] * t
            else:
                return history_det[-1], ll_algos["det"][0].phi
        if mode == "last":
            if "reverse" in method:
                return history_det[-t - 1 :], [ll_algos["det"][0].phi] * t
            else:
                return history_det[-1], ll_algos["det"][0].phi
        else:
            raise NotImplementedError

    print("Computing approximate deterministic hypergradients...")
    hg_results = defaultdict(lambda: defaultdict(list))
    eval_interval = max_iter_hg // n_eval_points
    for t in progressbar(range(0, max_iter_hg + 1)):
        # Compute hypergradient
        if t % eval_interval == 0 or t == max_iter_hg:
            for method, grad_f in (
                ("fixed", hg.fixed_point),
                ("reverse", hg.reverse),
                ("CG", hg.CG),
            ):
                wt, phi = get_wt_and_phi(t, mode, method)
                if method == "reverse":
                    grad = grad_f(wt, hparams, phi, ll_algos["det"][0].val_loss)
                else:
                    grad = grad_f(wt, hparams, t, phi, ll_algos["det"][0].val_loss)
                hg_results[method]["grad"].append(vectorize(grad))
                hg_results[method]['t'].append(t)
                hg_results[method]['n_samples'].append(t * ll_algos["det"][0].batch_size)


    if compute_stochastic_grads:
        def evaluate_stochastic_hgs(t, hg_results, print_interval=100):
            J_inner = math.ceil((t + 1) * J_inner_mult)

            phi_hat_map = lambda x, y: ll_algo_stoch.G_map(ll_algo_stoch.phi(x, y), y)
            phi_bar_map = lambda x, y: ll_algo_stoch.G_map(
                ll_algo_stoch.phi_bar_map(x, y, J_inner), y
            )

            T_bar_map = lambda x, y: ll_algo_stoch.phi_bar_map(x, y, J_inner)

            # get the last iterates for stochastic hypergradients
            if mode == "last_det":
                params_dec = history_det[-1]
                params_const = history_det[-1]
            elif mode == "last":
                params_dec = history_stoch_dec[-1]
                params_const = history_stoch_const[-1]
            elif mode == "normal":
                params_dec = history_stoch_dec[t]
                params_const = history_stoch_const[t]

            stoch_kwargs = dict(
                hparams=hparams,
                outer_loss=ll_algo_stoch.val_loss,
                K=t,
                print_interval=print_interval,
            )

            stoch_kwargs_dec = dict(params=params_dec, **stoch_kwargs)
            stoch_kwargs_const = dict(params=params_const, **stoch_kwargs)

            def add_grad_to_results(name, grad):
                print(f"adding results for {name}")
                hg_results[name]["grad"].append(vectorize([g.detach() for g in grad]))
                hg_results[name]["J_inner"].append(J_inner)
                hg_results[name]["t"].append(t)
                hg_results[name]["n_samples"].append((t + J_inner) * ll_algo_stoch.batch_size)

            
            # Compute NSID (decreasing step sizes)
            grad = NSID(
                T_hat_map=ll_algo_stoch.phi,
                T_bar_map=T_bar_map,
                G_map=ll_algo_stoch.G_map,
                step_size=step_size_stoch_dec,
                **stoch_kwargs_dec,
            )
            add_grad_to_results("fixed_stoch_dec", grad)
            
            # Compute NSID (constant step size)
            grad = NSID(
                T_hat_map=ll_algo_stoch.phi,
                T_bar_map=T_bar_map,
                G_map=ll_algo_stoch.G_map,
                step_size=step_size_ll,
                **stoch_kwargs_const,
            )
            add_grad_to_results("fixed_stoch_const", grad)
            
            # Compute SID as a special case of NSID (decreasing step sizes)
            grad = NSID(
                T_hat_map=phi_hat_map,
                T_bar_map=phi_bar_map,
                G_map=lambda x, y: x,
                step_size=step_size_stoch_dec,
                **stoch_kwargs_dec,
            )
            add_grad_to_results("fixed_stoch_dec_no_g", grad)

            # Compute SID as a special case of NSID (constant step size)
            grad = NSID(
                T_hat_map=phi_hat_map,
                T_bar_map=phi_bar_map,
                G_map=lambda x, y: x,
                step_size=step_size_ll,
                **stoch_kwargs_const,
            )
            add_grad_to_results("fixed_stoch_const_no_g", grad)

        print("Computing approximate stochastic hypergradients...")
        eval_interval = max_iter_hg_stoch // n_eval_points
        for t in progressbar(range(0, max_iter_hg_stoch + 1)):
            if t % eval_interval == 0 or t == max_iter_hg_stoch:
                evaluate_stochastic_hgs(
                    t, hg_results, print_interval=max(eval_interval // 10, 100)
                )

    for hg_name, res in hg_results.items():
        res[f"norm_diff"] = [
            torch.norm(h - true_hypergrad).cpu() for h in res["grad"]
        ]
        res[f"norm_diff_norm"] = [
            (torch.norm(h - true_hypergrad) / torch.norm(true_hypergrad)).cpu()
            for h in res["grad"]
        ]
        if "stoch" in hg_name:
            res["support_identification_index"] = ll_results["stoch_dec"][
                "support_identification_index"
            ]
        else:
            res["support_identification_index"] = ll_results["det"][
                "support_identification_index"
            ]

        hg_results[hg_name] = {k:v for k,v in res.items() if k!="grad"}

    if show_plots or save_plots:
        plot_hg_convergence(
            hg_results,
            metrics=("norm_diff", "norm_diff_norm"),
            xmetrics=("t", "n_samples"),
            show_plot=show_plots,
            save_plot=save_plots,
            save_path=save_path,
        )

    results = dict(
        ll_results=ll_results,
        hg_results=hg_results,
        params_true=params_true,
    )

    results = to_list(to_np(results))

    if save_path is not None:
        with open(join(save_path, "results.json"), "w") as json_file:
            json.dump(results, json_file, indent=2)
        if save_plots:
            recompute_plots(save_path)

    return results


def recompute_plots(path):
    with open(join(path, "conf.json"), "r") as json_file:
        conf = json.load(json_file)

    with open(join(path, "results.json"), "r") as json_file:
        results = json.load(json_file)

    results = to_torch(results)

    ll_results = results["ll_results"]
    hg_results = results["hg_results"]
    params_true = results["params_true"]

    plot_ll_convergence(
        ll_results,
        params_true,
        save_plot=True,
        show_plot=False,
        save_path=path,
    )

    plot_hg_convergence(
        hg_results,
        xmetrics=("t", "n_samples"),
        show_plot=False,
        save_plot=True,
        save_path=path,
    )


if __name__ == "__main__":
    main()
