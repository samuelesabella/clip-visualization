"""https://stackoverflow.com/questions/70628787/python-interactive-plotting-with-click-events"""
from __future__ import annotations
from pathlib import Path
from functools import partial
import typing as T

import plotly.graph_objects as go
from plotly.callbacks import Points
from plotly.graph_objs import Scatter, FigureWidget
import ipywidgets as widgets
from ipywidgets import Output, VBox
import PIL.Image as pil
from IPython.display import display, clear_output

from .clip_features import load_corpus, grep_images, get_features, get_2d_representation


def _get_scatter_plot(
    x,
    y,
    fig=None,
    color='#a3a7e4',
    name: T.Optional[str] = None,
    opacity: float = 1,
) -> T.Tuple[FigureWidget, Scatter]:
    if fig is None:
        fig = go.FigureWidget()
        fig.layout.width = 800  # type: ignore
        fig.layout.height = 600  # type: ignore
        fig.layout.hovermode = 'closest'  # type: ignore
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=name))
    fig.update_layout(template='simple_white')

    scatter = fig.data[-1]
    colors = [color] * len(x)
    scatter.marker.color = colors  # type: ignore
    scatter.marker.opacity = opacity  # type: ignore
    scatter.marker.size = [10] * 100  # type: ignore

    fig.update_traces(marker=dict(line=dict(color='DarkSlateGrey')))
    return fig, scatter


def _on_point_click(
    description,
    *args, **kwargs
):
    if isinstance(description, str):
        print(description)
        return
    display(pil.open(description))


def _get_figure(
    x: T.List[float],
    y: T.List[float],
    descriptions: T.List[T.Union[str, Path]],
    output: Output,
) -> FigureWidget:
    @output.capture(clear_output=True)
    def __on_point_click(scatter, point: Points, *args, target_descriptions, **kwargs):
        desc = target_descriptions[point.point_inds[0]]
        _on_point_click(desc, *args, **kwargs)

    fig, scatter = _get_scatter_plot(x, y)
    scatter.on_click(  # type: ignore
        partial(__on_point_click, target_descriptions=descriptions)
    )
    return fig


def get_clip_widget(
    x: T.List[float],
    y: T.List[float],
    descriptions: T.List[T.Union[str, Path]]
):
    out = widgets.Output(layout={'border': '1px solid black'})
    fig = _get_figure(x, y, descriptions, out)

    hbox_layout = widgets.Layout(
        display='flex',
        flex_flow='row',
        align_items='center',
        justify_content='flex-start',
        width='100%'
    )

    return widgets.HBox([fig, out], layout=hbox_layout)


def create_dataset_selection_vbox(clipwidget: ClipFeaturesLoader) -> VBox:
    dropdown_widgets = {
        'Images': get_clip_widget(clipwidget.images_x, clipwidget.images_y, clipwidget.images_description),
        'Corpus': get_clip_widget(clipwidget.corpus_x, clipwidget.corpus_y, clipwidget.corpus_description),
        'Images + Corpus': get_clip_widget(clipwidget.x, clipwidget.y, clipwidget.description),
    }

    dropdown = widgets.Dropdown(
        options=list(dropdown_widgets.keys()),
        value='Corpus',
        description='Select a dataset',
        disabled=False,
    )
    hbox_placeholder = widgets.Output()

    def dropdown_eventhandler(change):
        with hbox_placeholder:
            clear_output(wait=True)
            display(dropdown_widgets[change.new])

    dropdown.observe(dropdown_eventhandler, names='value')
    return VBox([dropdown, hbox_placeholder])


class ClipFeaturesLoader:
    def __init__(self) -> None:
        corpus = load_corpus('data/corpora.txt')
        images = grep_images('data/images/', recursive=True)

        self.corpus, self.corpus_description = get_features(corpus=corpus)
        self.corpus_x, self.corpus_y = get_2d_representation(self.corpus)

        self.images, self.images_description = get_features(images=images)
        self.images_x, self.images_y = get_2d_representation(self.images)

        self.data, self.description = get_features(corpus, images)
        self.x, self.y = get_2d_representation(self.data)
