import numpy as np
from bokeh.palettes import Category10
import sys
sys.path.append('../')
from utils.plotters import general_plot
from pathos.multiprocessing import ProcessingPool as Pool

from hurst import compute_Hc
from antropy import higuchi_fd
from nolds import dfa
from whittlehurst import whittle, variogram, tdml

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
epochs=100
batch_size=1000

models = {
    "Whittle": Model(workers, lambda seq: whittle(seq,"fGn_Paxson",K=10), take_diff=True),
    "TDML": Model(workers, tdml, take_diff=True),
    "Higuchi": Model(workers, lambda seq: 2-higuchi_fd(seq), take_diff=False),
    "Variogram": Model(workers, variogram, take_diff=False),
    "DFA": Model(workers, dfa, take_diff=True),
    "R/S": Model(workers, lambda seq: compute_Hc(seq, kind='change')[0], take_diff=True),
}

estimates = {}

def plot_volatility(vol, dates, name, fname, window=252, stride=63, labels=None):
    vol_tensor = np.lib.stride_tricks.sliding_window_view(np.array(vol, dtype=np.float32), window)[::stride]
    
    date_tensor = np.lib.stride_tricks.sliding_window_view(np.array(range(len(dates))), window)[::stride]
    measured_dates=[dates[x[len(x)-1]] for x in date_tensor]
    
    for nam, model in models.items():
        estimates[nam] = [float(val) for val in model(vol_tensor)]

    general_plot({
        "Ys": [vol] + list(estimates.values()),
        "Xs": [dates] + [measured_dates]*len(estimates),
        "xlabel": "Date",
        "ylabel": "Volatility and Hurst-estimate",
        "y_tick_step": 0.1,
        "title": "",
        "fname": fname,
        "dirname": "plots",
        "markers": None,#[None,".",".",".",".",".","."],
        "legend": {
            "location": "top_left",
            "labels": [name] + [f"{nam} Hurst" for nam in estimates.keys()]
        },
        "baselines":{
            "labels": [],
            "values": [0,0.5],
            "vertical": False,
            "colors": ["lightgrey"], # can be shorter than names
            "dashes": ["solid"] # can be shorter than namesself.colors
        },
        "matplotlib": {
            "calc_xtics": False,
            # "width": 8.1,
            # "height": 4.725,
            "width": 7, #8,
            "height": 5,
            "style": "default"
        },
        "colors":  ["grey"]+[Category10[10][i] for i in range(len(estimates))],
        #"colors":  ["grey"]+[Category10[10][i+2] for i in range(4)]+[Category10[10][0],Category10[10][1]],
        "dashes": ["solid","solid","dashed","dashdot","dotted","dotted","dotted"],
        "line45_color": None,
        "color_settings":{
            "bg_transparent": False
        }
    }, export_types=["png", "pdf", "json"])