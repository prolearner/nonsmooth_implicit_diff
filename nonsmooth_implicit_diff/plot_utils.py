import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from os.path import join
import numpy as np

from nonsmooth_implicit_diff.utils import vectorize


def save_show_plot(save_path, save_plot, show_plot):
    try:
        if save_plot:
            plt.savefig(save_path)
        if show_plot:
            plt.show()
    except ValueError:
        plt.yscale("linear")
        if save_plot:
            plt.savefig(save_path)
        if show_plot:
            plt.show()
    plt.close()


def plot_problem_features(X, y, w, show_plot=True, save_plot=False, save_path=None):
    # Plot the covariance matrix
    if show_plot or save_plot:
        plt.imshow(X.T @ X / X.shape[0], cmap="coolwarm")
        plt.colorbar()
        plt.title("Covariance Matrix of X")
        save_show_plot(
            join(save_path, "covariance.png"), save_plot=save_plot, show_plot=show_plot
        )

        # Plot the w
        if len(w.shape) == 1:
            plt.scatter(np.arange(len(w)), w)
            plt.title("w_true")
            save_show_plot(
                join(save_path, "w_true.png"), save_plot=save_plot, show_plot=show_plot
            )


def plot_w_true(w_approx, w_true, alpha=0.7):
    w_approx = w_approx.view(-1)
    w_true = w_true.view(-1)
    plt.scatter(np.arange(len(w_approx)), w_approx, label="w", alpha=alpha)
    plt.scatter(np.arange(len(w_true)), w_true, label="w_true", alpha=alpha)
    plt.title("w vs w_true")
    plt.yscale("log")
    plt.legend()


def plot_ll_convergence(
    ll_results,
    params_true,
    alpha=0.7,
    metrics=(
        "train_loss",
        "val_loss",
        "sparsity_frac",
        "diff_w",
        "diff_w_norm",
        "train_acc",
        "val_acc",
    ),
    show_plot=False,
    save_plot=False,
    save_path=None,
):
    for algo_name, res in ll_results.items():
        if "history" in res and params_true is not None:
            params_history = res["history"]
            w_history = [vectorize([p1.detach() for p1 in p]) for p in params_history]
            w_approx = w_history[-1]
            w_true = vectorize([p1.detach() for p1 in params_true])

            plot_w_true(w_approx.cpu(), w_true.cpu())
            save_show_plot(
                join(save_path, f"w_w_true_{algo_name}.png"),
                save_plot=save_plot,
                show_plot=show_plot,
            )

        plt.title("Losses during training")
        for l in ("train_loss", "val_loss"):
            plt.plot(res["t"], res[l], label=l)
        plt.yscale("log")
        plt.xlabel("# LL iterations")
        plt.legend()
        save_show_plot(
            join(save_path, f"losses_{algo_name}.png"),
            save_plot=save_plot,
            show_plot=show_plot,
        )

    for metric in metrics:
        if metric not in res:
            break
        for algo_name, res in ll_results.items():
            plt.title(metric)
            plt.plot(res["t"], res[metric], label=algo_name)

        s_index = res["support_identification_index"]

        if s_index is not None:
            plt.axvline(s_index, color="gray", linestyle="dashed")
        plt.legend()
        if metric != "sparsity_frac":
            plt.yscale("log")
        plt.xlabel("# LL iterations")
        save_show_plot(
            join(save_path, f"{metric}.png"), save_plot=save_plot, show_plot=show_plot
        )


def plot_hg_convergence(
    hypergrad_dict,
    metrics=("norm_diff", "norm_diff_norm"),
    xmetrics=("t", "n_samples"),
    show_plot=False,
    save_plot=False,
    save_path=None,
    methods_tuples=(
        ("fixed_stoch_const", "dotted"),
        ("fixed_stoch_dec", "dotted"),
        ("fixed_stoch_dec_no_g", "dotted"),
        ("fixed_stoch_const_no_g", "dotted"),
        ("fixed", "solid"),
        ("reverse", "dashed"),
        ("CG", "solid"),
        # ('reverse_fixed', 'dotted'),
    ),
):
    for mode in metrics:
        for xm in xmetrics:
            plt.title(mode)
            for hg_name, hg_style in methods_tuples:
                if hg_name not in hypergrad_dict.keys():
                    continue
                res = hypergrad_dict[hg_name]

                t = res[xm]
                diff = res[mode]
                support_index = res["support_identification_index"]
                plt.plot(t, diff, label=hg_name, linestyle=hg_style, marker="o")
            if bool(support_index) and xm == "t":
                plt.axvline(support_index, color="gray", linestyle="dashed")

            plt.xlabel(xm)
            plt.legend()
            plt.yscale("log")
            save_show_plot(
                join(save_path, f"hg_convergence_{mode}_{xm}.png"),
                save_plot=save_plot,
                show_plot=show_plot,
            )


def plot_heatmap(data_dict):
    # Convert dictionary to DataFrame
    df = pd.DataFrame(list(data_dict.items()), columns=["coords", "value"])
    df[["x", "y"]] = pd.DataFrame(df["coords"].tolist(), index=df.index)

    # Pivot the DataFrame to create a grid suitable for a heatmap
    heatmap_data = df.pivot_table(index="y", columns="x", values="value")
    heatmap_data.index = heatmap_data.index.astype(float).map(lambda x: f"{x:.2e}")
    heatmap_data.columns = heatmap_data.columns.astype(float).map(lambda x: f"{x:.2e}")

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, cmap="viridis", annot=True, xticklabels=True)


if __name__ == "__main__":
    # Example usage:
    data = {(-1, -2): 3, (0, 0): 5, (1, 2): 8, (2, -1): 4}
    plt(data)
