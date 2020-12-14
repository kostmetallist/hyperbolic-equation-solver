import plotly.graph_objects as go
import numpy as np


filename = 'dump/2020-12-14T01:28:14-14.csv'
values = np.genfromtxt(filename, delimiter=',')

fig = go.Figure(data=go.Volume(
        x=values[:, 0],
        y=values[:, 1],
        z=values[:, 2],
        value=values[:, 3],
        isomin=0,
        isomax=.01,
        opacity=.3,  # needs to be small to see through all surfaces
        surface_count=10,  # pick larger for good volume rendering
    )
)
fig.show()
