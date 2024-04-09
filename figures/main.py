import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def num_param_vs_acc(save_path):
    sns.set_theme(
        style="ticks",
        font="Times New Roman",
    )
    df = pd.read_csv('./data/num_param_vs_acc.csv')

    grid = sns.relplot(
        data=df,
        kind="line",
        x="# Params",
        y="Tiny-ImageNet Top-1 Accuracy (%)",
        hue="Scheme",
        style="Scheme",
        markers=True,
        height=4.5,
        aspect=4 / 4
    )
    ax = grid.axes[0, 0]
    for index, row in df.iterrows():
        x = row["# Params"]
        y = row["Tiny-ImageNet Top-1 Accuracy (%)"]
        text = row["Labels"]
        ax.text(x, y - 0.5, text, fontsize=9.5)

    grid.fig.tight_layout(pad=0.2, rect=(0, 0, 1, 1))

    if save_path:
        plt.savefig(save_path, format="pdf")
    else:
        plt.show()


def num_conv_kernel_vs_num_attn(save_path):
    sns.set_theme(
        style="ticks",
        font="Times New Roman",
    )
    df = pd.read_csv('./data/num_conv_kernel_vs_num_attn.csv')
    df = df.melt(id_vars=["# Conv2d Kernels"], var_name="# Attention Groups", value_name="CIFAR-10 Top-1 Acc (%)")

    grid = sns.catplot(
        data=df,
        kind="bar",
        x="# Conv2d Kernels",
        y="CIFAR-10 Top-1 Acc (%)",
        hue="# Attention Groups",
        height=2.5,
        aspect=16 / 9
    )
    grid.set(ylim=(70, 90))
    sns.move_legend(grid, "lower center", bbox_to_anchor=(.5, 0.75), ncol=4)
    grid.fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.9))

    if save_path:
        plt.savefig(save_path, format="pdf")
    else:
        plt.show()


def fix_num_conv_kernel_per_attn(save_path):
    sns.set_theme(
        style="ticks",
        font="Times New Roman",
    )
    df = pd.read_csv('./data/fix_num_conv_kernel_per_attn.csv')
    df = df.melt(id_vars=["# Conv2d Kernels per Attention Group"], var_name="# Attention Groups", value_name="CIFAR-10 Top-1 Acc (%)")

    grid = sns.catplot(
        data=df,
        kind="bar",
        x="# Conv2d Kernels per Attention Group",
        y="CIFAR-10 Top-1 Acc (%)",
        hue="# Attention Groups",
        height=2.5,
        aspect=16 / 9
    )
    grid.set(ylim=(70, 90))
    sns.move_legend(grid, "lower center", bbox_to_anchor=(.5, 0.75), ncol=4)
    grid.fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.9))

    if save_path:
        plt.savefig(save_path, format="pdf")
    else:
        plt.show()


def attn_group_vs_attn_depth(save_path):
    sns.set_theme(
        style="ticks",
        font="Times New Roman",
        palette="Paired"
    )
    df = pd.read_csv('./data/attn_group_vs_attn_depth.csv')
    df = df.melt(id_vars=["# Conv2d Kernels"], var_name="# Attention Groups - Attention Depth", value_name="CIFAR-10 Top-1 Acc (%)")

    grid = sns.catplot(
        data=df,
        kind="bar",
        x="# Conv2d Kernels",
        y="CIFAR-10 Top-1 Acc (%)",
        hue="# Attention Groups - Attention Depth",
        height=2.5,
        aspect=16 / 9
    )
    grid.set(ylim=(70, 90))
    sns.move_legend(grid, "lower center", bbox_to_anchor=(.5, 0.75), ncol=4)
    grid.fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.9))

    if save_path:
        plt.savefig(save_path, format="pdf")
    else:
        plt.show()


def conv_depth_vs_attn_depth(save_path):
    sns.set_theme(
        style="ticks",
        font="Times New Roman",
        palette="Paired"
    )
    df = pd.read_csv('./data/conv_depth_vs_attn_depth.csv')
    df = df.melt(id_vars=["# Conv2d Kernels"], var_name="Conv2d Depth - Attention Depth", value_name="CIFAR-10 Top-1 Acc (%)")

    grid = sns.catplot(
        data=df,
        kind="bar",
        x="# Conv2d Kernels",
        y="CIFAR-10 Top-1 Acc (%)",
        hue="Conv2d Depth - Attention Depth",
        height=2.5,
        aspect=16 / 9
    )
    grid.set(ylim=(68, 93))
    sns.move_legend(grid, "lower center", bbox_to_anchor=(.5, 0.75), ncol=4)
    grid.fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.9))

    if save_path:
        plt.savefig(save_path, format="pdf")
    else:
        plt.show()


def main():
    num_param_vs_acc("")
    # num_param_vs_acc("./output/num_param_vs_acc.pdf")
    # num_conv_kernel_vs_num_attn("")
    # num_conv_kernel_vs_num_attn("./output/num_conv_kernel_vs_num_attn.pdf")
    # fix_num_conv_kernel_per_attn("")
    # fix_num_conv_kernel_per_attn("./output/fix_num_conv_kernel_per_attn.pdf")
    # attn_group_vs_attn_depth("")
    # attn_group_vs_attn_depth("./output/attn_group_vs_attn_depth.pdf")
    # conv_depth_vs_attn_depth("")
    # conv_depth_vs_attn_depth("./output/conv_depth_vs_attn_depth.pdf")


if __name__ == "__main__":
    main()
