"""
Some ugly code for plotting stuff.
Not the main attraction, here for testing.
"""

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper, FreehandDrawTool, PolyDrawTool, CustomJS
from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.models.ranges import DataRange1d
from matplotlib import pyplot as plt
from matplotlib.colors import  to_rgb, to_hex
from matplotlib.patches import Patch
import numpy as np
import colorcet as cc
import os
import json
import yaml
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from functools import reduce
from typing import Optional
import random
import numbers

def deep_get(dictionary, keys, default):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

def flatten_list(xss):
    return [x for xs in xss for x in xs]
    #return list(np.array(xss).flatten())

def get_colors(nmbr):
    return cc.palette["glasbey_category10"][:nmbr]

def get_string_val(lst, i):
    try:
        return lst[i]
    except IndexError:
        return ''

def my_min(lst):
    return min([val for val in lst if val is not None])

def my_max(lst):
    return max([val for val in lst if val is not None])

def jitter(lst, width):
    return [x + np.random.uniform(low=-width, high=width) for x in lst]

def json_pretty_print(text, indent=4):
    level = 0
    list_level = 0
    inside_apostrophe = 0
    last_backslash_idx = -2
    ret = ""
    for i, c in enumerate(text):
        if c == "}" and inside_apostrophe % 2 == 0:
            level -= 1
            ret += "\n" + " " * (level*indent)
        ret += c
        if c == "{" and inside_apostrophe % 2 == 0:
            level += 1
            ret += "\n" + " " * (level*indent)
        elif c == "[" and inside_apostrophe % 2 == 0:
            list_level += 1
        elif c == "]" and inside_apostrophe % 2 == 0:
            list_level -= 1
        elif c == '"' and last_backslash_idx != i - 1:
            inside_apostrophe += 1
        elif c == "\\":
            last_backslash_idx = i
        elif c == "," and inside_apostrophe % 2 == 0 and list_level == 0:
            ret += "\n" + " " * (level*indent)
    return ret

def my_cmap(val, high, low=0, typ="log"):
    if val == 0:
        return "black"
    cmap = cc.CET_L8
    if val < 0:
        cmap = cc.CET_L6
        val = -val
    val -= low
    high -= low
    if typ == "log":
        val = math.log(val + 1)
        high = math.log(high + 1)
    elif typ == "sqrt":
        val = math.sqrt(val)
        high = math.sqrt(high)
    elif typ == "square":
        val = val**2
        high = high**2
    return cmap[math.ceil(val / high * (len(cmap) - 1))]

def darken_color(hex_color, factor=0.7):
    """Return a darker version of the given hex color."""
    rgb = np.array(to_rgb(hex_color))
    dark_hex = to_hex(rgb * factor)
    return dark_hex

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(data, fpath):
    with open(fpath, 'w') as outfile:
        #json.dump(data, outfile)#, indent=4)
        outfile.write(json_pretty_print(json.dumps(data, separators=(',', ': '), cls=NpEncoder)))

def save_yaml(data, fpath):
    with open(fpath, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def selet_not_None(list1, list2):
    return [val for i, val in enumerate(list1) if list2[i] != None]

def cyclic_fill(lst, length, default=None):
    if len(lst) == 0:
        lst = [default]
    return [lst[i % len(lst)] for i in range(length)]

matplotlib_dashes = {"solid": '-', "dotted": ':', "dashed": '--', "dotdash": '-.', "dashdot": '-.'}

def matplotlib_setcolors(ax, line_color="black", face_color="white", grid_color="0.9", **kwargs):
    ax.tick_params(axis='x', colors=line_color)
    ax.tick_params(axis='y', colors=line_color)
    ax.yaxis.label.set_color(line_color)
    ax.xaxis.label.set_color(line_color)
    ax.title.set_color(line_color)
    ax.set_facecolor(face_color)
    ax.patch.set_alpha(0)
    ax.spines['bottom'].set_color(line_color)
    ax.spines['left'].set_color(line_color)
    ax.spines['top'].set_color(grid_color)
    ax.spines['right'].set_color(grid_color)

def matlotlib_legend_loc(s):
    return s.replace("_", " ").replace("top", "upper").replace("bottom", "lower")

def init_matplotlib_figure(width=16, height=9, style="seaborn-poster", **kwargs):
    plt.style.use(style)
    fig = plt.figure(figsize=(width, height))
    ax = plt.subplot(111)
    return fig, ax

def init_matplotlib_grid_figure(width=16, height=9, style="seaborn-poster", grid_w=1, grid_h=1, grid_len=None, **kwargs):
    plt.style.use(style)
    fig, axs = plt.subplots(grid_h, grid_w, figsize=(grid_w * width, grid_h * height))
    if grid_len is not None:
        for i in range(grid_w):
            for j in range(grid_h):
                idx = j*grid_w + i
                if idx >= grid_len:
                    try:
                        fig.delaxes(axs[j][i])
                    except:
                        fig.delaxes(axs[i])
    return fig, axs.flatten()

def init_bokeh_figure(
    bokeh={
        "width": None,
        "height": None
    },
    title="",
    xlabel="",
    ylabel="",
    xscale="linear",
    yscale="linear",
    theme="caliber",
    p=None,
    yrange=None,
    **kwargs
):
    curdoc().theme = theme
    tools = "pan,box_zoom,wheel_zoom,xwheel_zoom,ywheel_zoom,save,reset"
    if p is None:
        fig_params = dict(title=title, tools=tools, x_axis_type=xscale, y_axis_type=yscale)
        if bokeh["width"] is None or bokeh["height"] is None:
            fig_params["sizing_mode"] = 'stretch_both'
        else:
            fig_params["plot_width"] = bokeh["width"]
            fig_params["plot_height"] = bokeh["height"]
        if not yrange is None:
            fig_params["y_range"] = DataRange1d(start=yrange[0], end=yrange[1], range_padding=0)  # type: ignore

        p = figure(**fig_params)  # type: ignore

    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel

    renderer = p.multi_line([[]], [[]], line_width=5, alpha=0.4, color='red')
    draw_tool_freehand = FreehandDrawTool(renderers=[renderer])
    draw_tool_poly = PolyDrawTool(renderers=[renderer])
    p.add_tools(draw_tool_freehand, draw_tool_poly)  # type: ignore

    return p

def bokeh_line45(p, min_x=0, max_x=1, line45_color=None, **kwargs):
    if line45_color is not None:
        src = ColumnDataSource(data=dict(x=[min_x, max_x], y=[min_x, max_x]))
        p.line("x", "y", line_color=line45_color, line_width=2, source=src, line_dash="dashed")
    return p

def matplotlib_line45(ax, min_x=0, max_x=1, line45_color=None, **kwargs):
    if line45_color is not None:
        ax.plot([min_x, max_x], [min_x, max_x], color=line45_color, linestyle="--", zorder=40)

class Plotter():
    def __init__(self, fname="", dirname="", neptune_experiment=None):
        self.neptune_experiment = neptune_experiment
        self.dirname = dirname
        self.fname = fname

    def get_full_path(self, extension="html", suffix="", make_subfolder=True):
        if suffix != "":
            suffix = f"_{suffix}"
        dirname = self.dirname
        if make_subfolder:
            dirname = os.path.join(dirname, extension)
        os.makedirs(dirname, exist_ok=True)
        return os.path.join(dirname, f"{self.fname}{suffix}.{extension}")

    def export_json(self, params, make_subfolder=True):
        params = params.copy()
        if "neptune_experiment" in params:
            del params["neptune_experiment"]
        try:
            full_path = self.get_full_path("json", make_subfolder=make_subfolder)
            save_json(params, full_path)
            if self.neptune_experiment is not None:
                self.neptune_experiment[f"json/{self.fname}"].upload(full_path)
                self.neptune_experiment.sync()
        except:
            full_path = self.get_full_path("yaml", make_subfolder=make_subfolder)
            save_yaml(params, full_path)
            if self.neptune_experiment is not None:
                self.neptune_experiment[f"yaml/{self.fname}"].upload(full_path)
                self.neptune_experiment.sync()

    def save_matplotlib_figure(self, dpi=240, bg_transparent=True, export_as=["pdf", "png"], make_subfolder=True):
        for extension in export_as:
            full_path = self.get_full_path(extension, make_subfolder=make_subfolder)
            plt.savefig(full_path, transparent=bg_transparent, dpi=dpi, bbox_inches='tight')
            if self.neptune_experiment is not None:
                self.neptune_experiment[f"{extension}/{self.fname}"].upload(full_path)
        if self.neptune_experiment is not None:
            self.neptune_experiment.sync()
        plt.close()

    def save_bokeh_figure(self, p, suffix="", make_subfolder=True):
        full_path = self.get_full_path("html", suffix, make_subfolder=make_subfolder)
        output_file(full_path)
        save(p)
        if self.neptune_experiment != None:
            self.neptune_experiment[f"html/{self.fname}{suffix}"].upload(full_path)
            self.neptune_experiment.sync()

class GeneralPlotter(Plotter):
    def __init__(
        self,
        Ys: list=[],
        Xs: list=[], # defaults to [1,2,3,4,...]
        errorbars=None,
        xlabel: str="",
        ylabel: str="",
        xscale: str="linear",
        yscale: str="linear",
        title: str="",
        colors=None, # the bulit in categorical colors go up to 256
        dashes=["solid"], #"solid", "dashed", "dotted", "dotdash", "dashdot"
        markers=["."],
        fname: str="general_plot",
        neptune_experiment=None,
        dirname: str="",
        line45_color=None, #None to turn off line45
        legend={
            "labels": [],
            "location": "top_right", # "top_left", "top_right", "bottom_left", "bottom_right", None
        },
        baselines={
            "labels": [],
            "values": [],
            "vertical": False,
            "colors": ["grey"], # can be shorter than names
            "dashes": ["dotted"] # can be shorter than namesself.colors
        },
        histogram=False,
        histogram_distr={
            "labels": [],
            "Xs": [],
            "colors": None, # can be shorter than names default: ["grey"]
            "bins": 100,
            "density": True,
            "alpha": 0.5
        },
        matplotlib={ # for png and svg
            "width": 16,
            "height": 9,
            "style": "seaborn-poster", # https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
            "png_dpi": 240, #use 240 for 4k resolution on 16x9 image
            "calc_xtics": False,
            "xtics": None
        },
        bokeh={
            "width": None, #set this to force bokeh plot to fixed dimensions (not recommended)
            "height": None #set this to force bokeh plot to fixed dimensions (not recommended)
        },
        color_settings={
            "suffix": "",
            "grid_color": "0.9",
            "face_color": "white",
            "line_color": "black",
            "bg_transparent": False
        }
    ):
        if len(Ys) > 0 and not isinstance(Ys[0], list):
            Ys = [Ys]

        if len(legend.get("labels", [])) + len(baselines.get("labels", [])) + len(histogram_distr.get("labels", [])) == 0:
            legend["location"] = None

        if legend.get("location", None) is None or len(legend.get("labels", [])) == 0:
            legend["labels"] = [None for _ in Ys]

        if legend.get("location", None) is None or len(baselines.get("labels", [])) == 0:
            baselines["labels"] = [None for _ in baselines["values"]]

        if legend.get("location", None) is None or len(histogram_distr.get("labels", [])) == 0:
            histogram_distr["labels"] = [None for _ in histogram_distr["Xs"]]

        if Xs == []:
            x_len = max([len(y) for y in Ys])
            Xs = [list(range(1, x_len + 1)) for _ in range(len(Ys))]
        else:
            if not isinstance(Xs[0], list):
                Xs = [Xs for _ in range(len(Ys))]

        if len(Xs) > 0:
            min_x = min(Xs[0])
            max_x = max(Xs[0])
            min_x = min(min([min(x) for x in Xs]), min_x)
            max_x = max(max([max(x) for x in Xs]), max_x)
        else:
            min_x = math.inf
            max_x = -math.inf

        if len(histogram_distr["Xs"]) > 0:
            min_x = min(min([min(x) for x in histogram_distr["Xs"]]), min_x)
            max_x = max(max([max(x) for x in histogram_distr["Xs"]]), max_x)

        if len(Ys) > 0:
            min_y = my_min(Ys[0])
            max_y = my_max(Ys[0])
            min_y = min(min([my_min(y) for y in Ys]), min_y)
            max_y = max(max([my_max(y) for y in Ys]), max_y)
        else:
            min_y = math.inf
            max_y = -math.inf

        if colors is None:
            colors = get_colors(len(Ys))
        else:
            colors = cyclic_fill(colors, len(Ys))
        dashes = cyclic_fill(dashes, len(Ys))
        if markers is None:
            markers = [""]
        markers = cyclic_fill(markers, len(Ys))

        if len(baselines["values"]) > 0:
            baselines["colors"] = cyclic_fill(baselines["colors"], len(baselines["values"]), "grey")
            baselines["dashes"] = cyclic_fill(baselines["dashes"], len(baselines["values"]), "dotted")

        if len(histogram_distr["Xs"]) > 0:
            if not histogram_distr["colors"] is None:
                histogram_distr["colors"] = cyclic_fill(histogram_distr["colors"], len(histogram_distr["Xs"]), "grey")
            else:
                histogram_distr["colors"] = get_colors(len(Ys) + len(histogram_distr["Xs"]))[len(Ys):]

        if not isinstance(histogram, list):
            histogram = [histogram for _ in range(len(Ys))]
        histogram = cyclic_fill(histogram, len(Ys), False)

        self.params = locals()
        super().__init__(fname, dirname, neptune_experiment)

    #########
    # bokeh #
    #########
    def make_bokeh_plot(self):
        p = init_bokeh_figure(**self.params)

        bokeh_line45(p, **self.params)

        sources = [
            ColumnDataSource(
                data=dict(
                    x=selet_not_None(x[:len(y)], y),
                    y=selet_not_None(y, y),
                    maxim=[max(selet_not_None(y, y)) for _ in selet_not_None(y, y)],
                    minim=[min(selet_not_None(y, y)) for _ in selet_not_None(y, y)],
                    argmax=[np.argmax(selet_not_None(y, y)) + 1 for _ in selet_not_None(y, y)],
                    argmin=[np.argmin(selet_not_None(y, y)) + 1 for _ in selet_not_None(y, y)],
                    label=[get_string_val(self.params["legend"]["labels"], i) for _ in selet_not_None(y, y)]
                )
            ) for i, (x, y) in enumerate(zip(self.params["Xs"], self.params["Ys"]))
        ]

        for (c, l, source, dash, marker, hist) in zip(
            self.params["colors"], self.params["legend"]["labels"], sources, self.params["dashes"],
            self.params["markers"], self.params["histogram"]
        ):
            if hist:
                if l != None:
                    p.vbar(top="y", source=source, color=c, legend_label=l)
                else:
                    p.vbar(top="y", source=source, color=c)

            else:
                if l != None:
                    p.line('x', 'y', color=c, line_width=2, source=source, line_dash=dash, legend_label=l)
                else:
                    p.line('x', 'y', color=c, line_width=2, source=source, line_dash=dash)
                if marker == ".":
                    p.circle('x', 'y', color=c, source=source)

        if self.params["errorbars"] is not None:
            for x, y, yerr, c in zip(
                self.params["Xs"], self.params["Ys"], self.params["errorbars"], self.params["colors"]
            ):
                err_source = ColumnDataSource(
                    data=dict(
                        error_low=[val - e for val, e in zip(selet_not_None(y, y), selet_not_None(yerr, yerr))],
                        error_high=[val + e for val, e in zip(selet_not_None(y, y), selet_not_None(yerr, yerr))],
                        x=selet_not_None(x[:len(y)], y)
                    )
                )

                p.segment(source=err_source, x0='x', y0='error_low', x1='x', y1='error_high', line_width=2, color=c)

        if len(self.params["histogram_distr"]["Xs"]) > 0:
            for y, color, label in zip(*[self.params["histogram_distr"][k] for k in ["Xs", "colors", "labels"]]):
                if self.params["xscale"] == "log":
                    bins = np.logspace(np.log10(min(y)), np.log10(max(y)), self.params["histogram_distr"]["bins"])
                else:
                    bins = self.params["histogram_distr"]["bins"]

                hist, edges = np.histogram(y, density=self.params["histogram_distr"]["density"], bins=bins)

                source = ColumnDataSource(
                    data=dict(
                        x=[(left+right) / 2 for left, right in zip(edges[:-1], edges[1:])],
                        y=hist,
                        left=edges[:-1],
                        right=edges[1:],
                        maxim=[max(hist) for _ in hist],
                        label=[label for _ in hist]
                    )
                )

                if self.params["yscale"] == "log":
                    bottom = 1
                else:
                    bottom = 0

                if label != None:
                    p.quad(
                        top="y",
                        bottom=bottom,
                        left="left",
                        right="right",
                        source=source,
                        fill_color=color,
                        line_color=color,
                        alpha=self.params["histogram_distr"]["alpha"],
                        legend_label=label
                    )
                else:
                    p.quad(
                        top="y",
                        bottom=bottom,
                        left="left",
                        right="right",
                        source=source,
                        fill_color=color,
                        line_color=color,
                        alpha=self.params["histogram_distr"]["alpha"]
                    )

        if len(self.params["baselines"]["values"]) > 0:
            for name, value, color, dash in zip(
                *[self.params["baselines"][k] for k in ["labels", "values", "colors", "dashes"]]
            ):
                if self.params["baselines"]["vertical"]:
                    try:
                        yrange = [self.params["min_y"] - 1, self.params["max_y"] + 1]
                    except:
                        yrange = [self.params["min_y"], self.params["max_y"]]
                    src = ColumnDataSource(
                        data={
                            "x": [value, value],
                            "y": yrange,
                            "maxim": [value, value],
                            "label": [name, name]
                        }
                    )
                else:
                    try:
                        xrange = [self.params["min_x"] - 1, self.params["max_x"] + 1]
                    except:
                        xrange = [self.params["min_x"], self.params["max_x"]]
                    src = ColumnDataSource(
                        data={
                            "x": xrange,
                            "y": [value, value],
                            "maxim": [value, value],
                            "label": [name, name]
                        }
                    )
                if name != None:
                    p.line("x", "y", line_dash=dash, line_color=color, line_width=2, source=src, legend_label=name)
                else:
                    p.line("x", "y", line_dash=dash, line_color=color, line_width=2, source=src)

        if self.params["legend"]["location"] != None:
            p.legend.location = self.params["legend"]["location"]

        tooltips = [
            (self.params["xlabel"], "@x"), (self.params["ylabel"], "@y"), ("name", "@label"), ("max", "@maxim"),
            ("argmax", "@argmax"), ("min", "@minim"), ("argmin", "@argmin")
        ]

        p.add_tools(HoverTool(tooltips=tooltips, mode='vline'))  # type: ignore
        p.add_tools(HoverTool(tooltips=tooltips))  # type: ignore
        return p

    # ##############
    # # matplotlib #
    # ##############
    def make_matplotlib_plot(self, ax):
        matplotlib = self.params["matplotlib"]

        ax.set_xlabel(self.params["xlabel"])
        ax.set_ylabel(self.params["ylabel"])
        ax.set_title(self.params["title"])

        ax.set_xscale(self.params["xscale"])
        ax.set_yscale(self.params["yscale"])

        matplotlib_line45(ax, **self.params)

        if len(self.params["baselines"]["values"]) > 0:
            for label, value, color, dash in zip(
                *[self.params["baselines"][k] for k in ["labels", "values", "colors", "dashes"]]
            ):
                if self.params["baselines"]["vertical"]:
                    yrange = [self.params["min_y"], self.params["max_y"]]
                    ax.plot([value, value], yrange, matplotlib_dashes[dash], label=label, color=color, zorder=20)
                else:
                    xrange = [self.params["min_x"], self.params["max_x"]]
                    ax.plot(xrange, [value, value], matplotlib_dashes[dash], label=label, color=color, zorder=20)

        for x, y, dash, color, label, marker, hist in zip(
            self.params["Xs"], self.params["Ys"], self.params["dashes"], self.params["colors"],
            self.params["legend"]["labels"], self.params["markers"], self.params["histogram"]
        ):
            if hist:
                plt.bar(x[:len(y)], y, color=color, label=label, zorder=30)
            else:
                ax.plot(x[:len(y)], y, matplotlib_dashes[dash], marker=marker, color=color, label=label, zorder=30)

        if self.params["errorbars"] is not None:
            for x, y, yerr in zip(self.params["Xs"], self.params["Ys"], self.params["errorbars"]):
                plt.errorbar(x, y, yerr=yerr, alpha=.5, fmt=':', capsize=3, capthick=1)
                data = {'x': x, 'y1': [val - e for val, e in zip(y, yerr)], 'y2': [val + e for val, e in zip(y, yerr)]}
                plt.fill_between(**data, alpha=.2)

        if len(self.params["histogram_distr"]["Xs"]) > 0:
            for y, color, label in zip(*[self.params["histogram_distr"][k] for k in ["Xs", "colors", "labels"]]):
                if self.params["xscale"] == "log":
                    bins = np.logspace(np.log10(min(y)), np.log10(max(y)), self.params["histogram_distr"]["bins"])
                else:
                    bins = self.params["histogram_distr"]["bins"]
                plt.hist(
                    y,
                    density=self.params["histogram_distr"]["density"],
                    bins=bins,
                    alpha=self.params["histogram_distr"]["alpha"],
                    color=color,
                    label=label,
                    zorder=10
                )

        if self.params["legend"]["location"] != None and (
            len(self.params["legend"]["labels"]) > 0 or len(self.params["baselines"]["labels"]) > 0 or
            len(self.params["histogram_distr"]["labels"]) > 0
        ):
            legend = ax.legend(framealpha=0.9, loc=matlotlib_legend_loc(self.params["legend"]["location"]))
            legend.set_zorder(40)
            frame = legend.get_frame()
            frame.set_facecolor(self.params["color_settings"].get("face_color", "white"))
            frame.set_edgecolor(self.params["color_settings"].get("grid_color", "0.9"))
            for text in legend.get_texts():
                text.set_color(self.params["color_settings"].get("line_color", "black"))

        ax.grid(True, color=self.params["color_settings"].get("grid_color", "0.9"), zorder=0)
        matplotlib_setcolors(ax, **self.params["color_settings"])

        if matplotlib.get("calc_xtics", False):
            x_max = max([len(str(x)) for x in x[:self.params["x_len"]]])
            ax.set_xticks(
                [
                    float(str(x[:self.params["x_len"]][i])[:6]) for i in range(self.params["x_len"])
                    if i % max(int(min(x_max, 6) * self.params["x_len"] / (4 * matplotlib["width"])), 1) == 0
                ]
            )
        if matplotlib.get("xtics", None) is not None:
            ax.set_xticks(matplotlib.get("xtics", None))

def general_plot(params, export_types=["json", "html", "png", "pdf"], make_subfolder=True):
    plotter = GeneralPlotter(**params)
    if "json" in export_types:
        plotter.export_json(params)
    if "html" in export_types:
        p = plotter.make_bokeh_plot()
        plotter.save_bokeh_figure(p)
    fig, ax = init_matplotlib_figure(**params.get("matplotlib", {"width": 16, "height": 9, "style": "seaborn-poster"}))
    plotter.make_matplotlib_plot(ax)
    plotter.save_matplotlib_figure(
        deep_get(params, "matplotlib.png_dpi", 240), deep_get(params, "color_settings.bg_transparent", False),
        [typ for typ in export_types if not typ in ["json", "html"]], make_subfolder
    )
    return fig

def general_grid_plot(
    params_list: list, width: int = 2, export_types=["json", "html", "png", "pdf"], make_subfolder=True
):
    if len(params_list) == 1:
        return general_plot(params_list[0], export_types=export_types)

    height = math.ceil(len(params_list) / width)
    fig, axs = init_matplotlib_grid_figure(
        grid_w=width,
        grid_h=height,
        **params_list[0].get("matplotlib", {
            "width": 16,
            "height": 9,
            "style": "seaborn-poster"
        })
    )
    bokeh_grid = [[None for _ in range(width)] for _ in range(height)]
    for idx, params in enumerate(params_list):
        plotter = GeneralPlotter(**params)
        if "html" in export_types:
            bokeh_grid[idx // width][idx % width] = plotter.make_bokeh_plot()  # type: ignore
        plotter.make_matplotlib_plot(axs[idx])
    if "json" in export_types:
        plotter.export_json(params_list)
    if "html" in export_types:
        grid = gridplot(
            bokeh_grid, sizing_mode='stretch_both'
        )  # type: ignore #, width=deep_get(params,"matplotlib.width",16)*width, height=deep_get(params,"matplotlib.height",9)*height)
        plotter.save_bokeh_figure(grid)
    plt.tight_layout()
    plotter.save_matplotlib_figure(
        deep_get(params, "matplotlib.png_dpi", 240), deep_get(params, "color_settings.bg_transparent", False),
        [typ for typ in export_types if not typ in ["json", "html"]], make_subfolder
    )
    return fig

class ScatterPlotter(Plotter):
    def __init__(
        self,
        Ys=[],
        Xs=[],
        xlabel: str="",
        ylabel: str="",
        xlim: list=[None, None],
        ylim: list=[None, None],
        xscale: str="linear",
        yscale: str="linear",
        title: str="",
        line45_color: str="red", #None to turn off line45
        colors=None, # the bulit in categorical colors go up to 256
        fname: str="general_plot",
        dirname: str="",
        neptune_experiment=None,
        circle_size: int=10,
        x_jitter: int=0,
        opacity: int=0,
        heatmap: bool=False,
        permute: bool=True,
        legend={
            "labels": [],
            "location": "bottom_right", # "top_left", "top_right", "bottom_left", "bottom_right", None
            "markerscale": 2.0
        },
        boundary={
            "functions": [],
            "dashes": ["dashed"], #"solid", "dashed", "dotted", "dotdash", "dashdot"
            "colors": ["red"], #"solid", "dashed", "dotted", "dotdash", "dashdot"
            "legend": [None],
        },
        baselines={
            "labels": [],
            "values": [],
            "vertical": False,
            "colors": ["grey"], # can be shorter than names
            "dashes": ["dotted"] # can be shorter than namesself.colors
        },
        matplotlib={ # for png and svg
            "width": 16,
            "height": 9,
            "style": "seaborn-poster", # https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
            "png_dpi": 240, #use 240 for 4k resolution on 16x9 image
        },
        bokeh={
            "width": None, #set this to force bokeh plot to fixed dimensions (not recommended)
            "height": None #set this to force bokeh plot to fixed dimensions (not recommended)
        },
        color_settings={
            "suffix": "",
            "grid_color": "0.9",
            "face_color": "white",
            "line_color": "black",
            "bg_transparent": False
        }
    ):
        if len(legend.get("labels", [])) == 0:
            legend["location"] = None

        if legend.get("location", "bottom_right") is None:
            legend["labels"] = []

        if hasattr(Xs, "tolist"):
            Xs = Xs.tolist()
        if hasattr(Ys, "tolist"):
            Ys = Ys.tolist()
        if isinstance(Xs[0], numbers.Number):
            Xs = [Xs]
        if isinstance(Ys[0], numbers.Number):
            Ys = [Ys]
        for i, x in enumerate(Xs):
            if hasattr(x, "tolist"):
                Xs[i] = x.tolist()
        for i, y in enumerate(Ys):
            if hasattr(y, "tolist"):
                Ys[i] = y.tolist()
        
        Xs = cyclic_fill(Xs, len(Ys))

        for x, y in zip(Xs, Ys):
            nan_idx = []
            for i in reversed(range(len(y))):
                if np.isnan(y[i]):
                    nan_idx.append(i)
            for idx in nan_idx:
                x.pop(idx)
                y.pop(idx)

        if colors is None:
            colors = get_colors(len(Ys))
        else:
            colors = cyclic_fill(colors, len(Ys))

        boundary["dashes"] = boundary.get("dashes", ["dashed"])
        boundary["dashes"] = cyclic_fill(boundary["dashes"], len(boundary["functions"]))
        boundary["colors"] = boundary.get("colors", ["red"])
        boundary["colors"] = cyclic_fill(boundary["colors"], len(boundary["functions"]))
        boundary["legend"] = boundary.get("legend", [None])
        boundary["legend"] = cyclic_fill(boundary["legend"], len(boundary["functions"]))

        if len(baselines["values"]) > 0:
            baselines["colors"] = cyclic_fill(baselines["colors"], len(baselines["values"]), "grey")
            baselines["dashes"] = cyclic_fill(baselines["dashes"], len(baselines["values"]), "dotted")

        min_x = math.inf
        max_x = -math.inf
        if len(Xs) > 0:
            min_x = min(min([min(X) for X in Xs]), min_x)
            max_x = max(max([max(X) for X in Xs]), max_x)

        min_y = math.inf
        max_y = -math.inf
        if len(Ys) > 0:
            min_y = min(min([min(y) for y in Ys]), min_y)
            max_y = max(max([max(y) for y in Ys]), max_y)

        self.params = locals()
        super().__init__(fname, dirname, neptune_experiment)

    #########
    # bokeh #
    #########
    def make_bokeh_plot(self, palette=cc.CET_L18, dot_limit: Optional[int] = None):
        p = init_bokeh_figure(**self.params)

        bokeh_line45(p, **self.params)

        if len(self.params["boundary"]["functions"]) > 0:
            x_range = np.linspace(self.params["min_x"], self.params["max_x"], 100)
            for bf, dash, color in zip(
                self.params["boundary"]["functions"], self.params["boundary"]["dashes"],
                self.params["boundary"]["colors"]
            ):
                src = ColumnDataSource(data=dict(
                    x=x_range,
                    y=[eval(bf) for x in x_range],
                ))
                p.line('x', 'y', color=color, line_width=2, source=src, line_dash=dash)
        if self.params["heatmap"]:
            for x, y, color, label in zip(
                self.params["Xs"], self.params["Ys"], self.params["colors"], self.params["legend"]["labels"]
            ):            
                xmin, xmax, ymin, ymax = self.get_limits(x, y)


                X_grid, Y_grid = np.mgrid[xmin:xmax:240j, ymin:ymax:240j]
                positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
                values = np.vstack([x, y])
                kernel = stats.gaussian_kde(values)
                Z = np.reshape(kernel(positions).T, X_grid.shape)

                p.image(
                    image=[np.transpose(Z)],
                    x=xmin,
                    y=ymin,
                    dw=xmax - xmin,
                    dh=ymax - ymin,
                    palette=palette,
                    level="image"
                )

                color_mapper = LinearColorMapper(palette=palette, low=0, high=np.amax(Z))
                color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0))

                p.add_layout(color_bar, 'right')  # type: ignore
        else:        
            all_x, all_y, all_c = [], [], []
            for x, y, color, label in zip(
                self.params["Xs"], self.params["Ys"], self.params["colors"], self.params["legend"]["labels"]
            ):
                all_x += x
                all_y += y
                all_c += [color for _ in x]

                p.scatter(
                    [x[0]],
                    [y[0]],
                    marker="circle",
                    size=self.params["circle_size"],
                    line_width=0,
                    color=color,
                    alpha=1 - self.params["opacity"],
                    legend_label=label
                )

            indices = list(range(len(all_x)))
            random.shuffle(indices)
            x_shuffled = [all_x[i] for i in indices]
            y_shuffled = [all_y[i] for i in indices]
            c_shuffled = [all_c[i] for i in indices]

            if dot_limit is not None:
                x_shuffled = x_shuffled[:dot_limit]
                y_shuffled = y_shuffled[:dot_limit]
                c_shuffled = c_shuffled[:dot_limit]
            
            p.scatter(
                jitter(x_shuffled, self.params["x_jitter"]),
                y_shuffled,
                marker="circle",
                size=self.params["circle_size"],
                line_width=0,
                color=c_shuffled,
                alpha=1 - self.params["opacity"]
            )            

            if not self.params["legend"].get("location", "bottom_right") is None:
                p.legend.location = self.params["legend"].get("location", "bottom_right")
                p.legend.glyph_height = int(self.params["circle_size"] * self.params["legend"].get("markerscale", 2.0) * 2)
                p.legend.glyph_width = int(self.params["circle_size"] * self.params["legend"].get("markerscale", 2.0) * 2)

        if len(self.params["baselines"]["values"]) > 0:
            for name, value, color, dash in zip(
                *[self.params["baselines"][k] for k in ["labels", "values", "colors", "dashes"]]
            ):
                if self.params["baselines"]["vertical"]:
                    src = ColumnDataSource(
                        data={
                            "x": [value, value],
                            "y": [self.params["min_y"] - 1, self.params["max_y"] + 1],
                            "maxim": [value, value],
                            "label": [name, name]
                        }
                    )
                else:
                    src = ColumnDataSource(
                        data={
                            "x": [self.params["min_x"] - 1, self.params["max_x"] + 1],
                            "y": [value, value],
                            "maxim": [value, value],
                            "label": [name, name]
                        }
                    )
                if name != None:
                    p.line("x", "y", line_dash=dash, line_color=color, line_width=2, source=src, legend_label=name)
                else:
                    p.line("x", "y", line_dash=dash, line_color=color, line_width=2, source=src)

        p.add_tools(HoverTool(tooltips=[(self.params["xlabel"], "$x"), (self.params["ylabel"], "$y")]))  # type: ignore

        code_hover = '''
            document.getElementsByClassName('bk-tooltip')[0].style.display = 'none';
            document.getElementsByClassName('bk-tooltip')[1].style.display = 'none';

        '''
        if self.params["heatmap"]:
            code_hover += "document.getElementsByClassName('bk-tooltip')[2].style.display = 'none';"
        p.hover.callback = CustomJS(code=code_hover)

        return p

    # ##############
    # # matplotlib #
    # ##############
    def make_matplotlib_plot(
        self,
        ax,
        color_settings={
            "suffix": "",
            "grid_color": "0.9",
            "face_color": "white",
            "line_color": "black",
            "bg_transparent": False,
            "cmap": "cet_CET_L18"
        },
        dot_limit: Optional[int] = None
    ):
        ax.set_xlabel(self.params["xlabel"])
        ax.set_ylabel(self.params["ylabel"])
        ax.set_title(self.params["title"])

        ax.set_xscale(self.params["xscale"])
        ax.set_yscale(self.params["yscale"])

        ax.set_xlim(self.params["xlim"])
        ax.set_ylim(self.params["ylim"])

        matplotlib_line45(ax, **self.params)

        if len(self.params["baselines"]["values"]) > 0:
            for label, value, color, dash in zip(
                *[self.params["baselines"][k] for k in ["labels", "values", "colors", "dashes"]]
            ):
                if self.params["baselines"]["vertical"]:
                    ax.plot(
                        [value, value], [self.params["min_y"] - 1, self.params["max_y"] + 1],
                        matplotlib_dashes[dash],
                        label=label,
                        color=color,
                        zorder=20
                    )
                else:
                    ax.plot(
                        [self.params["min_x"] - 1, self.params["max_x"] + 1], [value, value],
                        matplotlib_dashes[dash],
                        label=label,
                        color=color,
                        zorder=20
                    )

        if len(self.params["boundary"]["functions"]) > 0:
            x_range = np.linspace(self.params["min_x"], self.params["max_x"], 100)
            for bf, dash, color, label in zip(
                self.params["boundary"]["functions"], self.params["boundary"]["dashes"],
                self.params["boundary"]["colors"], self.params["boundary"]["legend"]
            ):
                plt.plot(
                    x_range, [eval(bf) for x in x_range], matplotlib_dashes[dash], color=color, zorder=45, label=label
                )

        plt.grid(True, color=color_settings["grid_color"], zorder=5, alpha=0.5)

        if self.params["heatmap"]:
            for x, y, color, label in zip(
                self.params["Xs"], self.params["Ys"], self.params["colors"], self.params["legend"]["labels"]
            ):
                xmin, xmax, ymin, ymax = self.get_limits(x, y)

                X_grid, Y_grid = np.mgrid[xmin:xmax:240j, ymin:ymax:240j]
                positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
                values = np.vstack([x, y])
                kernel = stats.gaussian_kde(values)
                Z = np.reshape(kernel(positions).T, X_grid.shape)

                im = ax.imshow(np.rot90(Z), cmap=color_settings["cmap"], extent=[xmin, xmax, ymin, ymax], zorder=0)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)

                cb = plt.colorbar(im, cax=cax)

                cb.ax.yaxis.set_tick_params(color=self.params["color_settings"].get("line_color", "black"))
                cb.outline.set_edgecolor(self.params["color_settings"].get("grid_color", "0.9"))
                plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=color_settings["line_color"])
        else:
            circle_size = self.params["circle_size"]
            if dot_limit is None:
                circle_size = circle_size / 2

            all_x, all_y, all_c = [], [], []
            for x, y, color, label in zip(
                self.params["Xs"], self.params["Ys"], self.params["colors"], self.params["legend"]["labels"]
            ):
                all_x += x
                all_y += y
                all_c += [color for _ in x]

                ax.scatter(
                    [x[0]], [y[0]],
                    marker='.',
                    color=color,
                    zorder=30,
                    alpha=1,
                    linewidth=0,
                    s=circle_size**2,
                    label=label
                )  # type: ignore

            xmin, xmax, ymin, ymax = self.get_limits(all_x, all_y)
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

            indices = list(range(len(all_x)))
            if self.params["permute"]:
                random.shuffle(indices)
            
            x_shuffled = [all_x[i] for i in indices]
            y_shuffled = [all_y[i] for i in indices]
            c_shuffled = [all_c[i] for i in indices]

            if dot_limit is not None:
                x_shuffled = x_shuffled[:dot_limit]
                y_shuffled = y_shuffled[:dot_limit]
                c_shuffled = c_shuffled[:dot_limit]

            ax.scatter(
                x_shuffled,
                y_shuffled,
                marker='.',
                color=c_shuffled,
                zorder=30,
                alpha=1 - self.params["opacity"],
                linewidth=0,
                s=circle_size**2,
                label=None
            )  # type: ignore

            if not self.params["legend"].get("location", "bottom_right") is None and len(
                self.params["legend"]["labels"]
            ) > 0:
                legend = ax.legend(
                    loc=matlotlib_legend_loc(self.params["legend"]["location"]),
                    markerscale=self.params["legend"].get("markerscale", 2.0)
                )
                legend.set_zorder(50)
                frame = legend.get_frame()
                frame.set_facecolor(self.params["color_settings"].get("face_color", "white"))
                frame.set_edgecolor(self.params["color_settings"].get("grid_color", "0.9"))
                for text in legend.get_texts():
                    text.set_color(self.params["color_settings"].get("line_color", "black"))

        matplotlib_setcolors(ax, **self.params["color_settings"])
    
    def get_limits(self, x, y):
        if self.params["xlim"][0] is None:
            xmin = min(x)
        else:
            xmin = self.params["xlim"][0]
        if self.params["xlim"][1] is None:
            xmax = max(x)
        else:
            xmax = self.params["xlim"][1]
        if self.params["ylim"][0] is None:
            ymin = min(y)
        else:
            ymin = self.params["ylim"][0]
        if self.params["ylim"][1] is None:
            ymax = max(y)
        else:
            ymax = self.params["ylim"][1]

        xrange = ymax-xmin
        yrange = ymax-ymin

        if self.params["xlim"][0] is None:
            xmin -= xrange*0.025
        if self.params["xlim"][1] is None:
            xmax += xrange*0.025
        if self.params["ylim"][0] is None:
            ymin -= yrange*0.025
        if self.params["ylim"][1] is None:
            ymax += yrange*0.025
        
        return xmin, xmax, ymin, ymax


def scatter_plot(params, export_types=["json", "html", "png", "pdf"], dot_limit: int = 20000, make_subfolder=True):
    plotter = ScatterPlotter(**params)

    if "json" in export_types:
        plotter.export_json(params, make_subfolder=make_subfolder)
    if "html" in export_types:
        p = plotter.make_bokeh_plot(dot_limit=dot_limit)
        plotter.save_bokeh_figure(p, make_subfolder=make_subfolder)

    for extension in export_types:
        if not extension in ["json", "html"]:
            fig, ax = init_matplotlib_figure(**plotter.params["matplotlib"])
            # PDF plotting with dot limit
            plotter.make_matplotlib_plot(ax, dot_limit=None if extension == "png" else dot_limit)
            plotter.save_matplotlib_figure(
                deep_get(params, "matplotlib.png_dpi", 240), deep_get(params, "color_settings.bg_transparent", False),
                [extension], make_subfolder
            )
    return fig

def scatter_grid_plot(
    params_list: list,
    width: int = 2,
    export_types=["json", "html", "png", "pdf"],
    dot_limit: int = 20000,
    make_subfolder=True,
    common_limits = True
):
    if len(params_list) == 1:
        return scatter_plot(params_list[0], export_types=export_types, dot_limit=dot_limit, make_subfolder=make_subfolder)
    
    if common_limits:
        xlim = [np.inf, -np.inf]
        ylim = [np.inf, -np.inf]
        for params in params_list:
            if isinstance(params["Xs"][0], numbers.Number):
                params["Xs"] = [params["Xs"]]
            for x in params["Xs"]:
                xlim[0] = min(min(x), xlim[0])
                xlim[1] = max(max(x), xlim[1])
            if isinstance(params["Ys"][0], numbers.Number):
                params["Ys"] = [params["Ys"]]
            for y in params["Ys"]:
                ylim[0] = min(min(y), ylim[0])
                ylim[1] = max(max(y), ylim[1])
        xgrow=(xlim[1]-xlim[0])*0.05
        xlim[0] -= xgrow
        xlim[1] += xgrow
        ygrow=(ylim[1]-ylim[0])*0.05
        ylim[0] -= ygrow
        ylim[1] += ygrow
        for params in params_list:
            params["xlim"] = xlim
            params["ylim"] = ylim
    
    dot_limit = int(dot_limit/len(params_list))

    height = math.ceil(len(params_list) / width)
    
    if "json" in export_types:
        plotter = ScatterPlotter(**params_list[0])
        plotter.export_json(params_list, make_subfolder=make_subfolder)

    for extension in export_types:
        if extension=="json":
            continue
        elif extension=="html":
            bokeh_grid = [[None for _ in range(width)] for _ in range(height)]
            for idx, params in enumerate(params_list):
                plotter = ScatterPlotter(**params)
                bokeh_grid[idx // width][idx % width] = plotter.make_bokeh_plot(dot_limit=dot_limit)  # type: ignore
            grid = gridplot(bokeh_grid, sizing_mode='stretch_both')  # type: ignore #, width=deep_get(params,"matplotlib.width",16)*width, height=deep_get(params,"matplotlib.height",9)*height)
            plotter.save_bokeh_figure(grid, make_subfolder=make_subfolder)
        else:
            fig, axs = init_matplotlib_grid_figure(
                grid_w=width,
                grid_h=height,
                grid_len=len(params_list),
                **params_list[0].get("matplotlib", {
                    "width": 16,
                    "height": 9,
                    "style": "seaborn-poster"
                })
            )
            for idx, params in enumerate(params_list):
                plotter = ScatterPlotter(**params)
                plotter.make_matplotlib_plot(axs[idx], dot_limit=None if extension == "png" else dot_limit)
            
            plt.tight_layout()
            plotter.save_matplotlib_figure(
                deep_get(params, "matplotlib.png_dpi", 240),
                deep_get(params, "color_settings.bg_transparent", False),
                [extension], make_subfolder
            )
    
    return fig


class AreaPlotter(Plotter):
    def __init__(
        self,
        X,
        Ys,
        fname:str = "spectrogram",
        dirname:str = "",
        title:str = "",
        xlabel:str = "x",
        ylabel:str = "y",
        normalized:bool = False,
        colors=None, # the bulit in categorical colors go up to 256
        legend_labels = None,
        measure:dir = {
            "Y": [],
            "label": None,
            "color": "black", # the bulit in categorical colors go up to 256
            "dash": "dashed"
        },
        matplotlib:dir = { # for png and svg
            "width": 16,
            "height": 9,
            "style": "seaborn-poster", #"seaborn-poster", "seaborn-talk"
            "png_dpi": 240 #use 240 for 4k resolution on 16x9 image
        },
        color_settings:dir = {
            "suffix": "",
            "grid_color": "0.9",
            "face_color": "white",
            "line_color": "black",
            "bg_transparent": False
        },
        neptune_experiment=None,
    ):
        if hasattr(X, "tolist"):
            X = X.tolist()
        if hasattr(Ys, "tolist"):
            Ys = Ys.tolist()
        if isinstance(Ys[0], numbers.Number):
            Ys = [Ys]
        for i, y in enumerate(Ys):
            if hasattr(y, "tolist"):
                Ys[i] = y.tolist()

        if colors is None:
            colors = get_colors(len(Ys))
        else:
            colors = cyclic_fill(colors, len(Ys))

        if len(measure["Y"]) > 0:
            measure["label"] = measure.get("label", None)
            measure["color"] = measure.get("color", "black")
            measure["dash"] = measure.get("dash", ["dashed"])

        self.params = locals()
        super().__init__(fname, dirname, neptune_experiment)

    ##############
    # matplotlib #
    ##############
    def make_matplotlib_plot(self, ax):
        ax.set_xlabel(self.params["xlabel"])
        ax.set_ylabel(self.params["ylabel"])

        x_array = np.array(self.params["X"])
        Y_array = np.array(self.params["Ys"]).T  # initially shape (num_series, N) now (N, num_series)
        Y_sum = np.sum(np.abs(Y_array), axis=1, keepdims=True)
        
        ax.set_xlim(np.min(x_array), np.max(x_array))
        ax.set_ylim(0, np.max(Y_sum)*1.05)

        if self.params["normalized"]:
            ax.set_ylim(0, 1)
            Y_sum[Y_sum == 0] = 1
            Y_array = Y_array / Y_sum

        Y_array_abs = np.abs(Y_array)

        # Plot the areas.
        cumulative = np.cumsum(Y_array_abs, axis=1)
        bottoms = np.hstack([np.zeros((Y_array_abs.shape[0], 1)), cumulative[:, :-1]])
        signs = np.sign(Y_array)
        for j in range(len(self.params["Ys"])):
            pos_color = self.params["colors"][j]
            neg_color = darken_color(pos_color, 0.7)
            for i in range(len(x_array) - 1):
                seg_color = pos_color if signs[i, j] > 0 else neg_color
                ax.fill_between(
                    x_array[i:i + 2],
                    [bottoms[i, j], bottoms[i + 1, j]],
                    [cumulative[i, j], cumulative[i + 1, j]],
                    color=seg_color,
                    alpha=0.7
                )

        ax.set_title(self.params["title"], pad=50)

        legend_handles = []
        if self.params["legend_labels"] is not None:
            for label, color in zip(self.params["legend_labels"], self.params["colors"]):
                legend_handles.append(Patch(facecolor=color, label=label))

        if len(self.params["measure"]["Y"]) > 0:
            dash = matplotlib_dashes[self.params["measure"]["dash"]]

            ax2 = ax.twinx()
            ax2.plot(
                x_array,
                self.params["measure"]["Y"],
                dash,
                color=self.params["measure"]["color"],
                label=self.params["measure"]["label"],
                linewidth=2
            )
            ax2.set_ylabel(self.params["measure"]["label"])
            legend_handles.append(plt.Line2D(
                [0], [0], color=self.params["measure"]["color"],
                linestyle=dash, label=self.params["measure"]["label"]
            ))
        
        ax.legend(
            legend_handles, [h.get_label() for h in legend_handles],
            ncol=len(legend_handles),
            bbox_to_anchor=(0, 1),
            loc='lower left'
        )

        matplotlib_setcolors(ax, **self.params["color_settings"])      

def area_plot(params, export_types=["json", "png", "pdf"], make_subfolder=True):
    plotter = AreaPlotter(**params)

    if "json" in export_types:
        plotter.export_json(params, make_subfolder)

    fig, ax = init_matplotlib_figure(**plotter.params["matplotlib"])
    plotter.make_matplotlib_plot(ax)
    plotter.save_matplotlib_figure(
        deep_get(params, "matplotlib.png_dpi", 240),
        deep_get(params, "color_settings.bg_transparent", False),
        [typ for typ in export_types if typ != "json"],
        make_subfolder
    )
    return fig

def area_grid_plot(params_list, width: int = 2, export_types=["json", "png", "pdf"], make_subfolder=True):
    if len(params_list) == 1:
        return area_plot(params_list[0], export_types=export_types, make_subfolder=make_subfolder)
    
    height = math.ceil(len(params_list) / width)

    if "json" in export_types:
        plotter = AreaPlotter(**params_list[0])
        plotter.export_json(params_list, make_subfolder=make_subfolder)

    fig, axs = init_matplotlib_grid_figure(
        grid_w=width,
        grid_h=height,
        grid_len=len(params_list),
        **params_list[0].get("matplotlib", {
            "width": 16,
            "height": 9,
            "style": "seaborn-poster"
        })
    )

    for idx, params in enumerate(params_list):
        plotter = AreaPlotter(**params)
        plotter.make_matplotlib_plot(axs[idx])
    
    plt.tight_layout()

    plotter.save_matplotlib_figure(
        deep_get(params, "matplotlib.png_dpi", 240),
        deep_get(params, "color_settings.bg_transparent", False),
        [typ for typ in export_types if typ != "json"],
        make_subfolder
    )
    return fig

class SpectrumPlotter(Plotter):
    def __init__(
        self,
        spectrum: list,
        fname:str = "spectrogram",
        dirname:str = "",
        title:str = "",
        xlabel:str = "Window",
        ylabel:str = "Spectrum",
        xscale:str = "linear",
        yscale:str = "linear",
        matplotlib:dir = { # for png and svg
            "width": 16,
            "height": 9,
            "style": "seaborn-poster", #"seaborn-poster", "seaborn-talk"
            "png_dpi": 240 #use 240 for 4k resolution on 16x9 image
        },
        neptune_experiment=None,
    ):
        if dirname != "" and not os.path.isdir(dirname):
            os.makedirs(dirname)
        self.params = locals()
        super().__init__(fname, dirname, neptune_experiment)

    ##############
    # matplotlib #
    ##############
    def make_matplotlib_plot(self, ax, spectrum):
        return ax.imshow(spectrum, cmap="viridis", origin="lower", aspect="auto")

def spectrum_plot(params, export_types=["json", "png", "pdf"]):
    plotter = SpectrumPlotter(**params)
    spectrum = plotter.params["spectrum"]

    if "json" in export_types:
        plotter.export_json(params)

    fig, ax = init_matplotlib_figure(**plotter.params["matplotlib"])
    plotter.make_matplotlib_plot(ax, spectrum)
    plotter.save_matplotlib_figure(
        deep_get(params, "matplotlib.png_dpi", 240),
        deep_get(params, "color_settings.bg_transparent", False),
        [typ for typ in export_types if typ != "json"]
    )
    return fig