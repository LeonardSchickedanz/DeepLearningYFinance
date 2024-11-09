import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
#import data/data.py as d
#import data as d

def plot_losses(losses, test_losses):
    fig = go.Figure()

    # Training Loss
    fig.add_trace(go.Scatter(
        y=losses,
        mode='lines',
        name='Training Loss',
        line=dict(color='blue')
    ))

    # Test Loss
    fig.add_trace(go.Scatter(
        y=test_losses,
        mode='lines',
        name='Test Loss',
        line=dict(color='orange')
    ))

    # Layout
    fig.update_layout(
        title='Training and Test Loss Over Time',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend=dict(x=0.85, y=0.95),
        template="plotly_white"
    )

    # Grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Show plot
    fig.show()


#def plot_stocks(data):

x = np.random.random(50)
y = np.random.random(50)

fig = px.scatter(x,y)
fig.show()
