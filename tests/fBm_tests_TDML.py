import numpy as np
from bokeh.palettes import Category10
from pathos.multiprocessing import ProcessingPool as Pool
from time import time

from whittlehurst import fbm_gen, tdml

from utils.metrics import calc_dev, calc_rmse
from utils.plotters import general_plot, scatter_grid_plot


class Model:
    def __init__(self, num_cores=1, estimator=tdml, take_diff=True):
        self.num_cores = num_cores
        self.take_diff = take_diff
        self.estimator = estimator

    def __call__(self, x):
        x = np.asarray(x)
        if self.take_diff:
            x = x[:, :-1] - x[:, 1:]
        with Pool(self.num_cores) as p:
            est = p.map(self.estimator, x)
        return est


sample_count = 10000
workers = 32
dashes = ["solid", "dashed"]

models = {
    "TDML_jit": Model(workers, lambda seq: tdml(seq, jit=True), take_diff=True),
    "TDML_classic": Model(workers, lambda seq: tdml(seq, jit=False), take_diff=True),
}

plot_names = list(models.keys())
totals = {name: [] for name in plot_names}
RMSEs = {name: [] for name in plot_names}

n_s = [256, 1024, 4096]
plot_dir = "./plots/fBm_TDML"
palette = Category10[10]

for n in n_s:
    print(f"n={n}")

    if n > 10000:
        sample_count = 1000
    elif n > 5000:
        sample_count = 3000
    elif n > 2500:
        sample_count = 6000
    else:
        sample_count = 10000

    est = {name: [] for name in plot_names}
    for name in plot_names:
        totals[name].append(0.0)

    orig, inputs = fbm_gen(sample_count, n=n, threads=workers)

    keys = list(models.keys())
    np.random.shuffle(keys)

    for name in keys:
        start = time()
        est[name] += list(models[name](inputs))
        totals[name][-1] += (time() - start) * workers / sample_count

    plot_keys = list(est.keys())
    accumulated_n = n_s[:len(next(iter(totals.values())))]

    x_range, deviations_lst, biases_lst, deviation_aucs, bias_aucs = calc_dev(
        [orig] * len(plot_keys),
        list(est.values()),
        0, 1, 1000, 0.05,
    )

    general_plot(
        {
            "Ys": biases_lst,
            "Xs": x_range,
            "xlabel": "H",
            "ylabel": "Local Bias",
            "title": "",
            "fname": f"fBm_TDML_{n:05d}_biases",
            "dirname": plot_dir,
            "markers": None,
            "baselines": {
                "labels": [],
                "values": [0],
                "vertical": False,
                "colors": ["grey"],
                "dashes": ["solid"],
            },
            "legend": {
                "location": "best",
                "labels": [f"{name} (AUC={auc:.4f})" for name, auc in zip(plot_keys, bias_aucs)],
            },
            "dashes": dashes,
            "matplotlib": {
                "calc_xtics": False,
                "width": 6,
                "height": 4,
                "style": "default",
            },
            "color_settings": {
                "bg_transparent": False,
            },
        },
        export_types=["png", "pdf", "json"],
    )

    general_plot(
        {
            "Ys": deviations_lst,
            "Xs": x_range,
            "xlabel": "H",
            "ylabel": "Local Deviation",
            "title": "",
            "fname": f"fBm_TDML_{n:05d}_deviations",
            "dirname": plot_dir,
            "markers": None,
            "baselines": {
                "labels": [],
                "values": [0],
                "vertical": False,
                "colors": ["grey"],
                "dashes": ["solid"],
            },
            "legend": {
                "location": "best",
                "labels": [f"{name} (AUC={auc:.4f})" for name, auc in zip(plot_keys, deviation_aucs)],
            },
            "dashes": dashes,
            "matplotlib": {
                "calc_xtics": False,
                "width": 6,
                "height": 4,
                "style": "default",
            },
            "color_settings": {
                "bg_transparent": False,
            },
        },
        export_types=["png", "pdf", "json"],
    )

    x_range, rmse_lst, global_rmse = calc_rmse(
        [orig] * len(plot_keys),
        list(est.values()),
        0, 1, 1000, 0.05,
    )

    for i, name in enumerate(plot_keys):
        RMSEs[name].append(global_rmse[i])

    general_plot(
        {
            "Ys": rmse_lst,
            "Xs": x_range,
            "xlabel": "H",
            "ylabel": "Local RMSE",
            "title": "",
            "fname": f"fBm_TDML_{n:05d}_RMSE",
            "dirname": plot_dir,
            "markers": None,
            "baselines": {
                "labels": [],
                "values": [0],
                "vertical": False,
                "colors": ["grey"],
                "dashes": ["solid"],
            },
            "legend": {
                "location": "best",
                "labels": [f"{name} (RMSE={rmse:.4f})" for name, rmse in zip(plot_keys, global_rmse)],
            },
            "dashes": dashes,
            "matplotlib": {
                "calc_xtics": False,
                "width": 6,
                "height": 4,
                "style": "default",
            },
            "color_settings": {
                "bg_transparent": False,
            },
        },
        export_types=["png", "pdf", "json"],
    )

    scatter_grid = [
        {
            "Xs": orig,
            "Ys": Ys,
            "xlabel": "Real H",
            "ylabel": "Inferred H",
            "fname": f"fBm_TDML_{n:05d}_scatter_grid",
            "dirname": plot_dir,
            "circle_size": 10,
            "opacity": 0.3,
            "colors": [palette[i % len(palette)]],
            "line45_color": "black",
            "legend": {
                "location": "best",
                "labels": [f"{name}\nRMSE:{global_rmse[i]:.4f}\nbias:{bias_aucs[i]:.4f}\ndev:{deviation_aucs[i]:.4f}"],
                "markerscale": 2.0,
            },
            "matplotlib": {
                "width": 5,
                "height": 5,
                "style": "default",
            },
        }
        for i, (name, Ys) in enumerate(est.items())
    ]
    scatter_grid_plot(
        params_list=scatter_grid,
        width=3,
        export_types=["png", "pdf", "json"],
        make_subfolder=True,
        common_limits=True,
    )

    scatter_grid = [
        {
            "Xs": orig,
            "Ys": [y - x for x, y in zip(orig, Ys)],
            "xlabel": "H",
            "ylabel": "Error",
            "fname": f"fBm_TDML_{n:05d}_scatter_grid_error",
            "dirname": plot_dir,
            "circle_size": 10,
            "opacity": 0.3,
            "colors": [palette[i % len(palette)]],
            "line45_color": None,
            "baselines": {
                "labels": [None],
                "values": [0],
                "vertical": False,
                "colors": ["black"],
                "dashes": ["dashed"],
            },
            "legend": {
                "location": "best",
                "labels": [f"{name}\nRMSE:{global_rmse[i]:.4f}\nbias:{bias_aucs[i]:.4f}\ndev:{deviation_aucs[i]:.4f}"],
                "markerscale": 2.0,
            },
            "matplotlib": {
                "width": 5,
                "height": 5,
                "style": "default",
            },
        }
        for i, (name, Ys) in enumerate(est.items())
    ]
    scatter_grid_plot(
        params_list=scatter_grid,
        width=3,
        export_types=["png", "pdf", "json"],
        make_subfolder=True,
        common_limits=True,
    )

    total_keys = list(totals.keys())
    total_vals = list(totals.values())

    general_plot(
        {
            "Ys": total_vals,
            "Xs": accumulated_n,
            "xlabel": "Sequence Length",
            "ylabel": "Calculation Time (s)",
            "xscale": "log2",
            "yscale": "log",
            "title": "",
            "fname": "fBm_TDML_calc_times",
            "dirname": plot_dir,
            "markers": None,
            "legend": {
                "location": "best",
                "labels": total_keys,
            },
            "dashes": dashes,
            "matplotlib": {
                "calc_xtics": False,
                "width": 6,
                "height": 4,
                "style": "default",
            },
            "color_settings": {
                "bg_transparent": False,
            },
        },
        export_types=["png", "pdf", "json"],
    )

    rmse_keys = list(RMSEs.keys())
    rmse_vals = list(RMSEs.values())

    general_plot(
        {
            "Ys": rmse_vals,
            "Xs": accumulated_n,
            "xlabel": "Sequence Length",
            "ylabel": "RMSE",
            "xscale": "log2",
            "yscale": "log",
            "title": "",
            "fname": "fBm_TDML_RMSE",
            "dirname": plot_dir,
            "markers": None,
            "legend": {
                "location": "best",
                "labels": rmse_keys,
            },
            "dashes": dashes,
            "matplotlib": {
                "calc_xtics": False,
                "width": 6,
                "height": 4,
                "style": "default",
            },
            "color_settings": {
                "bg_transparent": False,
            },
        },
        export_types=["png", "pdf", "json"],
    )

    prices = np.array(rmse_vals) * np.array(total_vals)
    general_plot(
        {
            "Ys": prices.tolist(),
            "Xs": accumulated_n,
            "xlabel": "Sequence Length",
            "ylabel": "RMSE * Calculation Time",
            "xscale": "log2",
            "yscale": "log",
            "title": "",
            "fname": "fBm_TDML_RMSE_compute",
            "dirname": plot_dir,
            "markers": None,
            "legend": {
                "location": "best",
                "labels": rmse_keys,
            },
            "dashes": dashes,
            "matplotlib": {
                "calc_xtics": False,
                "width": 6,
                "height": 4,
                "style": "default",
            },
            "color_settings": {
                "bg_transparent": False,
            },
        },
        export_types=["png", "pdf", "json"],
    )