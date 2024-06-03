from hypergradient_convergence import run_elastic_net
import traceback

EXP_DIR = "exps/data_poisoning_stochastic"


def main():
    run()
    # for seed in range(10):
    #         run(random_state=seed,
    #             n_eval_points=100,
    #             save_path=EXP_DIR)

def run(
    random_state=1,
    compute_stochastic_grads=True,
    poisoned_frac=.3,
    alpha_l1=0.02,
    alpha_l2=.1,
    beta_mult=2,
    gamma_mult=0.001,
    n_samples=60000,
    batch_size_ll=300,
    max_iter_true=10000,
    max_iter_ll=1000,
    max_iter_ll_stoch=20000,
    max_iter_hg=1000,
    max_iter_hg_stoch=20000,
    J_inner_mult = 0.05,
    opt_L_mu_div = 10.0,
    n_eval_points=20,
    **kwargs,
):
    conf = {k: v for k, v in locals().copy().items() if k!='kwargs'}
    run_base(**conf, **kwargs)


def run_base(
    compute_stochastic_grads=True,
    poisoned_frac=.5,
    problem_type="poisoning",
    mode="last_det",  #  normal last last_det
    data="mnist",  # mnist, synth
    random_state=1,
    n_samples=60000,
    shuffle=True,
    val_size=0.5,
    alpha_l1=0.1,
    alpha_l2=0.1,
    max_iter_ll=4000,
    max_iter_ll_stoch=2000,
    max_iter_hg=400,
    max_iter_hg_stoch=2000,
    step_size_phi="optimal",
    opt_L_mu_div = 10.0,
    beta_ll="optimal",
    gamma_ll="optimal",
    gamma_mult=1.,
    beta_mult=1.,
    batch_size_ll=300,
    eval_inteval_ll=100,
    n_eval_points=5,
    J_inner_mult = 1/20,
    max_iter_true=40000,
    show_plots=False,
    save_plots=True,
    save_path=EXP_DIR,
    device="cuda:0",
):
    conf = locals().copy()
    run_elastic_net(**conf)
    
if __name__ == "__main__":
    main()