import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
from fsrs_optimizer import power_forgetting_curve
from scipy.optimize import minimize


def cum_concat(x):
    return list(accumulate(x))


def create_time_series(df):
    df = df[(df["delta_t"] != 0) & (df["rating"].isin([1, 2, 3, 4]))].copy()
    df["i"] = df.groupby("card_id").cumcount() + 1
    t_history = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[i] for i in x]), include_groups=False
    )
    r_history = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x]), include_groups=False
    )
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history for item in sublist
    ]
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    df.dropna(inplace=True)
    return df[(df["delta_t"] > 0) & (df["i"] == 2)].sort_values(by=["review_th"])


def fit_stability(delta_t, retention, size):
    def loss(stability):
        y_pred = power_forgetting_curve(delta_t, stability)
        logloss = sum(
            -(retention * np.log(y_pred) + (1 - retention) * np.log(1 - y_pred))
            * np.sqrt(size)
        )
        return logloss

    res = minimize(loss, x0=1, bounds=[(0.01, 100)])
    return res.x[0]


def outlier_analysis(df, outlier_filter_fn):
    for r_history in ("1", "2", "3", "4"):
        original_group = df[df["r_history"] == r_history]
        filtered_group = outlier_filter_fn(original_group.copy())
        fig = plt.figure(figsize=(16, 10))
        for i, group in enumerate((original_group, filtered_group)):
            if group.empty:
                continue
            sub_group = (
                group.groupby("delta_t").agg({"y": ["mean", "count"]}).reset_index()
            )
            delta_t = sub_group["delta_t"]
            retention = sub_group[("y", "mean")]
            size = sub_group[("y", "count")]
            stability = fit_stability(delta_t, retention, size)
            total_cnt = sum(sub_group[("y", "count")])
            ax1 = fig.add_subplot(2, 2, i + 1)
            ax1.bar(delta_t, size, color="#ff7f0e", ec="k", linewidth=0, label="Size")
            ax1.set_ylim(0, max(size) * 1.1)
            # ax1.semilogy()
            ax1.set_ylabel("Size")
            ax2 = ax1.twinx()
            ax2.plot(delta_t, retention, color="#1f77b4", marker="*", label="Retention")
            ax2.plot(
                delta_t,
                power_forgetting_curve(delta_t, stability),
                color="red",
                label=f"s={stability:.2f}",
            )
            ax2.set_ylim(0, 1.1)
            ax2.set_title(f"first rating: {r_history}, total: {total_cnt}")
            ax2.set_xlabel("delta_t")
            ax2.set_ylabel("Retention")
            ax2.grid(True)

            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles = handles1 + handles2
            labels = labels1 + labels2

            ax2.legend(handles, labels, loc="upper right")
        plt.show()
