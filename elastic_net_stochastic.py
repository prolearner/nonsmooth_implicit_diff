from hypergradient_convergence import run_elastic_net

from os.path import join

EXP_DIR = "exps/elastic_net_stochastic"

def main():
    run(random_state=1)
    # for seed in range(10):
    #         run(random_state=seed,
    #             n_eval_points=100,
    #             save_path=EXP_DIR)

def run(
    mode="last_det", 
    n_samples=50000,
    n_eval_points=100,
    alpha_l1=0.1,
    alpha_l2=0.01,
    noise=0.01,
    max_iter_true=20000,
    max_iter_ll=400,
    max_iter_ll_stoch=8000,
    max_iter_hg=160,
    max_iter_hg_stoch=8000,
    beta_ll="optimal",
    gamma_ll="optimal",
    batch_size_ll=100,
    beta_mult=.5,
    gamma_mult=2.,
    J_inner_mult=1,
    **kwargs,
):  
    conf = {k: v for k, v in locals().copy().items() if k!='kwargs'}
    run_base(**conf, **kwargs)


def run_base(
    problem_type="HPO",
    mode="last_det",  #  normal last last_det
    data="synth",
    random_state=4,
    n_samples=50000,
    n_features=100,
    noise=0.01,
    effective_rank=30,
    n_informative=30,
    correlated=True,
    shuffle=False,
    val_size=0.4,
    test_size=0.4,
    #  alpha_l1=0.0017782794100389228, alpha_l2=1e-5,
    alpha_l1=0.001,
    alpha_l2=0.001,
    max_iter_ll=1000,
    max_iter_ll_stoch=12000,
    max_iter_hg=400,
    max_iter_hg_stoch=12000,
    max_iter_true=20000,
    step_size_phi="optimal",
    beta_ll="optimal",
    gamma_ll="optimal",
    # beta_ll=1,
    # gamma_ll=100,
    beta_mult=0.7,
    gamma_mult=1.4,
    J_inner_mult=1,
    batch_size_ll=100,
    eval_inteval_ll=100,
    show_plots=False,
    save_plots=True,
    save_path=EXP_DIR,
    device="cpu",
    compute_stochastic_grads=True,
    **kwargs,
):  
    conf = {k: v for k, v in locals().copy().items() if k!='kwargs'}
    run_elastic_net(**conf, **kwargs)
    
if __name__ == "__main__":
    main()
