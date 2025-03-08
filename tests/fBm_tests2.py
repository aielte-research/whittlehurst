import numpy as  np
import random
from bokeh.palettes import Category10
from tqdm import trange
from pathos.multiprocessing import ProcessingPool as Pool
import time

from hurst import compute_Hc
from antropy import higuchi_fd

from whittlehurst import whittle, variogram, fbm

from utils.metrics import calc_dev, calc_rmse
from utils.plotters import general_plot, scatter_grid_plot


class Model():
    def __init__(self, num_cores=1, estimator=whittle, take_diff=True):
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

workers=32
epochs=10
batch_size=100

models = dict(
    Whittle=Model(workers, lambda seq: whittle(seq, "fGn"), take_diff=True),
    Variogram=Model(workers, lambda seq: variogram(seq), take_diff=False),
    Higuchi=Model(workers, lambda seq: 2-higuchi_fd(seq), take_diff=False),
    R_over_S=Model(workers, lambda seq: compute_Hc(seq, kind='change')[0], take_diff=True)
)

totals = {nam: [] for nam in models.keys()}
RMSEs = [[] for _ in models]

n_s = [200,400,800,1600,3200,6400,12800,25600]
for n in n_s:
    print(f"n={n}")
    orig = []
    est = {nam: [] for nam in models.keys()}
    for nam in totals.keys():
        totals[nam].append(0.0)

    pbar=trange(epochs)
    for _ in pbar:
        pbar.set_description("Generating...")
        inputs = []
        for _ in range(batch_size):
            H = random.uniform(0, 1)
            orig.append(H)
            process = fbm(H, n)
            inputs.append(np.asarray(process))

        for nam, model in models.items():
            start = time.time()
            pbar.set_description(nam)
            est[nam] += list(model(inputs))
            totals[nam][-1] += time.time() - start

    for nam in models.keys():
        totals[nam][-1] /= epochs*batch_size/workers

    x_range, deviations_lst, biases_lst, deviation_aucs, bias_aucs = calc_dev(
        [orig]*len(models),
        list(est.values()),
        0, 1, 1000, 0.05
    )

    general_plot({
        "Ys": biases_lst,
        "Xs": x_range,
        "xlabel": "Hurst",
        "ylabel": "Local Bias",
        "title": "",
        "fname": f"fBm_Hurst_{n}_biases",
        "dirname": "./plots/fBm_estimators",
        "markers": None,
        "legend": {
            "location": "bottom_right",
            "labels": list(models.keys())
        },
        "dashes": ["solid","dashed","dashdot","dotted"],
        "matplotlib": {
            "calc_xtics": False,
            "width": 6,
            "height": 4,
            "style": "default"
        },
        "color_settings": {
            "bg_transparent": False
        }
    }, export_types=["png", "pdf"])

    general_plot({
        "Ys": deviations_lst,
        "Xs": x_range,
        "xlabel": "Hurst",
        "ylabel": "Local Deviation",
        "title": "",
        "fname": f"fBm_Hurst_{n}_deviations",
        "dirname": "./plots/fBm_estimators",
        "markers": None,
        "legend": {
            "location": "bottom_right" if n<1600  else "top_right",
            "labels": list(models.keys())
        },
        "dashes": ["solid","dashed","dashdot","dotted"],
        "matplotlib": {
            "calc_xtics": False,
            "width": 6,
            "height": 4,
            "style": "default"
        },
        "color_settings": {
            "bg_transparent": False
        }
    }, export_types=["png", "pdf"])

    x_range, rmse_lst, global_rmse = calc_rmse(
        [orig]*len(models),
        list(est.values()),
        0, 1, 1000, 0.05
    )

    for i, rmse in enumerate(global_rmse):
        RMSEs[i].append(rmse)

    general_plot({
        "Ys": rmse_lst,
        "Xs": x_range,
        "xlabel": "Hurst",
        "ylabel": "Local RMSE",
        "title": "",
        "fname": f"fBm_Hurst_{n}_RMSE",
        "dirname": "./plots/fBm_estimators",
        "markers": None,
        "legend": {
            "location": "top_right",
            "labels": [f"{nam} RMSE={rmse:.4f}" for nam, rmse in zip(models.keys(),global_rmse)]
        },
        "dashes": ["solid","dashed","dashdot","dotted"],
        "matplotlib": {
            "calc_xtics": False,
            "width": 6,
            "height": 4,
            "style": "default"
        },
        "color_settings": {
            "bg_transparent": False
        }
    }, export_types=["png", "pdf"])

    scatter_grid = [{
        "Xs": orig,
        "Ys": Ys,
        "xlabel": "real H",
        "ylabel": "inferred H",
        #"title": title,
        "fname": f"fBm_Hurst_{n}_scatter_grid",
        "dirname": "./plots/fBm_estimators",
        "circle_size": 10,
        "opacity": 0.3,
        "colors": [Category10[10][i]],
        "line45_color": "black",
        "legend": {
            "location": "bottom_right",
            "labels": [nam],
            "markerscale": 2.0
        },
        "matplotlib": {
            "width": 6,
            "height": 6,
            "style": "default"
        }
    } for i, (nam, Ys) in enumerate(est.items())]
    scatter_grid_plot(
        params_list=scatter_grid,
        width=2,
        export_types=["png", "pdf"],
        make_subfolder=True,
        common_limits=True
    )

    scatter_grid = [{
        "Xs": orig,
        "Ys": [y-x for x, y in zip(orig,Ys)],
        "xlabel": "H",
        "ylabel": "Error",
        #"title": title,
        "fname": f"fBm_Hurst_{n}_error_scatter_grid",
        "dirname": "./plots/fBm_estimators",
        "circle_size": 10,
        "opacity": 0.3,
        "colors": [Category10[10][i]],
        "line45_color": None,
        "legend": {
            "location": "bottom_right",
            "labels": [nam],
            "markerscale": 2.0
        },
        "matplotlib": {
            "width": 6,
            "height": 6,
            "style": "default"
        }
    } for i, (nam, Ys) in enumerate(est.items())]
    scatter_grid_plot(
        params_list=scatter_grid,
        width=2,
        export_types=["png", "pdf"],
        make_subfolder=True,
        common_limits=True
    )

general_plot({
    "Ys": list(totals.values()),
    "Xs": n_s,
    "xlabel": "Sequence Length",
    "ylabel": "Calculation Time (s)",
    "xscale": "log",
    "yscale": "log",
    "title": "",
    "fname": f"fBm_Hurst_calc_times",
    "dirname": "./plots/fBm_estimators",
    "markers": None,
    "legend": {
        "location": "top_left",
        "labels": list(totals.keys())
    },
    "dashes": ["solid","dashed","dashdot","dotted"],
    "matplotlib": {
        "calc_xtics": False,
        "width": 6,
        "height": 4,
        "style": "default"
    },
    "color_settings": {
        "bg_transparent": False
    }
}, export_types=["png", "pdf"])

general_plot({
    "Ys": RMSEs,
    "Xs": n_s,
    "xlabel": "sequence length (n)",
    "ylabel": "RMSE",
    "xscale": "log",
    "yscale": "log",
    "title": "",
    "fname": f"fBm_Hurst_RMSE",
    "dirname": "./plots/fBm_estimators",
    "markers": None,
    "legend": {
        "location": "bottom_left",
        "labels": list(models.keys())
    },
    "dashes": ["solid","dashed","dashdot","dotted"],
    "matplotlib": {
        "calc_xtics": False,
        "width": 6,
        "height": 4,
        "style": "default"
    },
    "color_settings": {
        "bg_transparent": False
    }
}, export_types=["png", "pdf"])

prices = np.array(RMSEs)*np.array(list(totals.values()))
general_plot({
    "Ys": prices.tolist(),
    "Xs": n_s,
    "xlabel": "sequence length (n)",
    "ylabel": "RMSE * Calculation Time",
    "xscale": "log",
    "yscale": "log",
    "title": "",
    "fname": f"fBm_Hurst_RMSE_compute",
    "dirname": "./plots/fBm_estimators",
    "markers": None,
    "legend": {
        "location": "top_left",
        "labels": list(models.keys())
    },
    "dashes": ["solid","dashed","dashdot","dotted"],
    "matplotlib": {
        "calc_xtics": False,
        "width": 6,
        "height": 4,
        "style": "default"
    },
    "color_settings": {
        "bg_transparent": False
    }
}, export_types=["png", "pdf"])
