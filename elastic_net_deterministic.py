from hypergradient_convergence import run_elastic_net


from os.path import join

EXP_DIR = "exps/elastic_net_deterministic"


def main():
    run(alpha_l1=0.02, alpha_l2=0.002)
    run(alpha_l1=0.002, alpha_l2=0.002, max_iter_hg=1000)



def run(
    problem_type="HPO",
    mode="normal",
    data="synth",
    random_state=3,
    n_samples=500,
    n_features=100,
    noise=0.1,
    effective_rank=30,
    n_informative=30,
    correlated=False,
    shuffle=False,
    val_size=0.4,
    test_size=0.4,
    #  alpha_l1=0.0017782794100389228, alpha_l2=1e-5,
    alpha_l1=0.01,
    alpha_l2=0.002,
    max_iter_ll=2000,
    max_iter_hg=500,
    step_size_phi="optimal",
    eval_inteval_ll=100,
    n_eval_points=200,
    max_iter_true=10000,
    show_plots=False,
    save_plots=True,
    save_path=EXP_DIR,
    device="cpu",
    compute_stochastic_grads=False,
    **kwargs,
):  
    conf = {k: v for k, v in locals().copy().items() if k!='kwargs'}
    run_elastic_net(**conf, **kwargs)
    
if __name__ == "__main__":
    main()
