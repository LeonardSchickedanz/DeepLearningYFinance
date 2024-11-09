import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


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

import plotly.graph_objects as go
import pandas as pd

def plot_stocks(datum, wert1, wert2):
    # Erstelle den Plot
    fig = go.Figure()

    # Hinzufügen der ersten Datenreihe als Linie
    fig.add_trace(go.Scatter(
        x=datum,
        y=wert1,
        mode='lines',
        name='Stock Price 1',
        line=dict(color='blue')  # Farbe für die erste Reihe
    ))

    # Hinzufügen der zweiten Datenreihe als Linie
    fig.add_trace(go.Scatter(
        x=datum,
        y=wert2,
        mode='lines',
        name='Stock Price 2',
        line=dict(color='red')  # Farbe für die zweite Reihe
    ))

    # Layout anpassen
    fig.update_layout(
        title='Stock Prices Over Time',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )

    # Achsenoptionen und Gitterlinien anpassen
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Plot anzeigen
    fig.show()

# Methode aufrufen
d_time_series = pd.read_excel('../data_xlsx/d_timeseries_raw.xlsx', index_col=0)
datum=d_time_series.index
wert = d_time_series['4. close']
wert2 = wert + 100
plot_stocks(datum, wert, wert2)

