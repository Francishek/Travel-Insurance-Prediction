import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple
from scipy.stats import chi2_contingency
import phik


def bootstrap_median_diff(
    data1, data2, num_iterations=1000, ci=95
) -> Tuple[float, float]:
    """
    Computes the bootstrap confidence interval for the difference in medians
    between two datasets.

    This function randomly resamples the two input datasets with replacement
    to simulate the sampling distribution of the difference in medians. It
    calculates the confidence interval for the difference based on the
    bootstrap resamples.

    Parameters:
    -----------
    data1 : array-like
        First dataset.
    data2 : array-like
        Second dataset.
    num_iterations : int, optional (default=1000)
        Number of bootstrap iterations to perform.
    ci : float, optional (default=95)
        The confidence level for the interval (e.g., 95 for a 95% confidence interval).

    Returns:
    --------
    lower : float
        The lower bound of the bootstrap confidence interval.
    upper : float
        The upper bound of the bootstrap confidence interval.
    """
    boot_diffs = []
    n1 = len(data1)
    n2 = len(data2)
    for i in range(num_iterations):
        boot_sample1 = np.random.choice(data1, size=n1, replace=True)
        boot_sample2 = np.random.choice(data2, size=n2, replace=True)
        boot_diffs.append(np.median(boot_sample1) - np.median(boot_sample2))
    lower = np.percentile(boot_diffs, (100 - ci) / 2)
    upper = np.percentile(boot_diffs, 100 - (100 - ci) / 2)
    return lower, upper


def crosstab_chi2_test(df: pd.DataFrame, col_x: str, col_y: str) -> None:
    """
    Computes and prints a normalized crosstab and performs Chi-squared test of independence.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data
        col_x (str): The name of the row variable (e.g., 'FrequentFlyer')
        col_y (str): The name of the column variable (e.g., 'TravelInsurance')
    """
    print(f"\n Crosstab of {col_x} vs. {col_y}\n")

    cross_tab_norm = pd.crosstab(df[col_x], df[col_y], normalize="index").round(2) * 100
    print("Normalized Crosstab (%):")
    print(cross_tab_norm)

    cross_tab_counts = pd.crosstab(df[col_x], df[col_y])
    chi2, p, dof, expected = chi2_contingency(cross_tab_counts)

    print("\nChi-squared test results:")
    print(f"Chi-squared Statistic: {chi2:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p:.4f}")

    if p < 0.05:
        print(" Statistically significant association (p < 0.05)")
    else:
        print(" No significant association (p ≥ 0.05)")


def plot_distribution_numerical(
    data: pd.DataFrame,
    column: str,
    target: str = "TravelInsurance",
    figsize: Tuple[int, int] = (8, 6),
    bins: int = 10,
) -> None:
    """
    Plot comparative analysis of a numerical feature vs target variable
    including general and conditional distributions and boxplots.

    Parameters:
    - data: Input DataFrame
    - column: Numerical column to analyze
    - target: Target variable name (default: TravelInsurance)
    - figsize: Figure size (width, height)
    - bins: Number of histogram bins (default: 10)
    Plot comparative analysis of numerical feature vs target variable.
    """
    col_data = data[column]
    plot_data = data.copy()
    plot_data["_col_"] = col_data

    plt.figure(figsize=figsize)

    plt.subplot(2, 2, 1)
    sns.boxplot(
        y=plot_data["_col_"],
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    plt.title(f"General Boxplot of {column}")
    plt.xlabel("")
    plt.ylabel(column)

    # Boxplot by target
    plt.subplot(2, 2, 2)
    sns.boxplot(
        x=target,
        y="_col_",
        data=plot_data,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    plt.title(f"{column.capitalize()} by {target}")
    plt.xlabel(target)
    plt.ylabel(column)

    plt.subplot(2, 2, 3)
    sns.histplot(plot_data["_col_"].dropna(), bins=bins, kde=True, color="skyblue")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 4)
    sns.histplot(
        data=plot_data,
        x="_col_",
        hue=target,
        bins=bins,
        kde=True,
        element="step",
        common_norm=False,
    )
    plt.title(f"{column.capitalize()} distribution by {target}")
    plt.xlabel(column)
    plt.ylabel("Density")

    plt.tight_layout()
    plt.show()


def stacked_bar_with_percent(
    data: pd.DataFrame,
    column_x: str,
    column_y: str = "TravelInsurance",
    figsize: Tuple[int, int] = (8, 4),
) -> None:
    """
    Plot stacked bar chart with percentages for binary target analysis.

    Parameters:
    - data: Input DataFrame
    - column_x: Categorical feature to analyze
    - column_y: Target variable (default: TravelInsurance)
    - figsize: Figure size (width, height)
    """
    cross = pd.crosstab(data[column_x], data[column_y], normalize="index") * 100

    ax = cross.plot.bar(stacked=True, figsize=figsize, mark_right=True)

    for patch in ax.patches:
        height = patch.get_height()
        if height < 1:
            continue
        ax.annotate(
            f"{height:.0f}%",
            (patch.get_x() + patch.get_width() / 2, patch.get_y() + height / 2),
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )

    plt.title(f"Insurance Purchase Rate by {column_x}", pad=15)
    plt.xlabel(column_x)
    plt.ylabel("Percentage of Customers")
    plt.legend(
        title="Purchased Insurance", labels=["No", "Yes"], bbox_to_anchor=(1.05, 1)
    )
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def countplot_with_percent(
    data: pd.DataFrame,
    column: str,
    orientation: str = "vertical",
    figsize: Tuple[int, int] = (4, 4),
) -> None:
    """
    Plot a countplot with percentages, with optional horizontal orientation.

    Parameters:
    - data: Input DataFrame
    - column: Column to plot
    - orientation: "vertical" (default) or "horizontal"
    - figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    if orientation == "horizontal":
        ax = sns.countplot(y=column, data=data)
    else:
        ax = sns.countplot(x=column, data=data)

    total = len(data)

    for p in ax.patches:
        if orientation == "horizontal":
            count = int(p.get_width())
            x = p.get_width() + 0.01
            y = p.get_y() + p.get_height() / 2
            ha, va = "left", "center"
        else:
            count = int(p.get_height())
            x = p.get_x() + p.get_width() / 2
            y = p.get_height() + 0.01
            ha, va = "center", "bottom"

        percentage = 100 * count / total
        ax.annotate(
            f"{percentage:.1f}%",
            (x, y),
            ha=ha,
            va=va,
            fontsize=10,
        )

    plt.title(f"Distribution of {column}")
    if orientation == "horizontal":
        plt.xlabel("Count")
        plt.ylabel(column)
    else:
        plt.xlabel(column)
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def plot_distribution_with_kde(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    figsize: Tuple[int, int] = (8, 4),
    kde: bool = True,
) -> None:
    """
    Plots the distribution of a numerical column with optional KDE.

    Parameters:
    - df: pandas DataFrame
    - column: str, name of the column to plot
    - bins: int, number of bins in histogram
    - figsize: tuple, size of the plot
    - kde: bool, whether to include KDE plot
    """

    plt.figure(figsize=figsize)
    sns.histplot(data=df, x=column, kde=kde, bins=bins)

    plt.title(f"Distribution of {column.title()} with KDE")
    plt.xlabel(column.title())
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_scatter_with_reg_and_lowess(
    df: pd.DataFrame,
    x: str,
    y: str = "quality",
    figsize: Tuple[int, int] = (8, 4),
    alpha: float = 0.5,
) -> None:
    """
    Plots scatterplot with both linear regression and LOWESS smoothing lines.

    Parameters:
    - df: pandas DataFrame
    - x: str, feature column
    - y: str, target column
    - figsize: tuple, size of the plot
    - alpha: float, transparency of scatter points
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x, y=y, alpha=alpha)

    sns.regplot(
        data=df,
        x=x,
        y=y,
        scatter=False,
        line_kws={"color": "blue"},
        label="Linear Regression",
    )

    sns.regplot(
        data=df,
        x=x,
        y=y,
        scatter=False,
        lowess=True,
        line_kws={"color": "red"},
        label="LOWESS Smoother",
    )

    plt.title(f"{x.title()} vs. {y.title()} with Linear & LOWESS Lines")
    plt.xlabel(x.title())
    plt.ylabel(y.title())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_violin_by_quality(
    df: pd.DataFrame,
    feature: str,
    target: str = "quality",
    figsize: Tuple[int, int] = (8, 4),
) -> None:
    """
    Plots a violin plot of a feature grouped by target, with one overall mean and median line.


    Parameters:
    - df: pandas DataFrame
    - feature: str, the feature to plot on the y-axis
    - target: str, the grouping variable (default is "quality")
    - figsize: tuple, size of the plot
    """
    plt.figure(figsize=figsize)
    sns.violinplot(x=target, y=feature, data=df)

    mean_val = df[feature].mean()
    median_val = df[feature].median()

    plt.axhline(mean_val, color="red", linestyle="--", label=f"Mean ({mean_val:.2f})")
    plt.axhline(
        median_val, color="black", linestyle=":", label=f"Median ({median_val:.2f})"
    )

    plt.title(f"{feature.title()} by wine {target}")
    plt.xlabel(target.title())
    plt.ylabel(feature.title())
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_phik_correlation_heatmap(
    df: pd.DataFrame,
    figsize: tuple = (10, 6),
) -> None:
    """
    Generates and displays a heatmap of the Phi-k correlation matrix
    for a given DataFrame. Only the lower triangle of the matrix is displayed

    Args:
        df (pd.DataFrame): The input DataFrame for which to calculate
                           and visualize the Phi-k correlation.
        figsize (tuple, optional): A tuple (width, height) in inches
                                   for the figure size of the heatmap.
                                   Defaults to (10, 8).

    Returns:
        None: Displays the heatmap plot.
    """
    try:
        phik_corr = df.phik_matrix()
    except Exception as e:
        print(f"Error calculating Phi-k matrix: {e}")
        print(
            "Please ensure your DataFrame is suitable for phik_matrix() "
            "and the 'phik' library is correctly installed."
        )
        return

    mask = np.triu(np.ones_like(phik_corr, dtype=bool))

    plt.figure(figsize=figsize)

    sns.heatmap(
        phik_corr,
        mask=mask,
        annot=True,
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Phi-k Correlation (Lower triangle)", fontsize=16)
    plt.tight_layout()
    plt.show()
