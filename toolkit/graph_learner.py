import plotly.plotly as py
import plotly.graph_objs as go

py.offline.init_notebook_mode(connected=True)

py.offline.iplot({
    "data": [Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
    "layout": Layout(title="Perceptron")
})
