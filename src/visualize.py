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
import numpy as np

def plot_stocks(dates, y_test, y_pred_test, scaler):

    if len(y_test) != len(y_pred_test):
        print(f"y_test ({len(y_test)}) & y_pred_test ({len(y_pred_test)}) do not have the same length")
    if scaler is None:
        print("scaler is none")

    y_test = np.array(y_test)
    y_pred_test = np.array(y_pred_test)

    # descaling
    y_test_transformed = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_transformed = scaler.inverse_transform(y_pred_test.reshape(-1, 1))

    # inverse
   # y_test_transformed = y_test_transformed[::-1]
   # y_pred_transformed = y_pred_transformed[::-1]

    min_len = min(len(dates), len(y_test_transformed), len(y_pred_transformed))
    dates = dates[-min_len:]
    y_test_transformed = y_test_transformed[-min_len:]
    y_pred_transformed = y_pred_transformed[-min_len:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_test_transformed.flatten(),
        mode='lines',
        name='Tatsächliche Werte',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_pred_transformed.flatten(),
        mode='lines',
        name='Vorhersagen',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='Aktienkursprognose: Vergleich tatsächliche Werte vs. Vorhersagen',
        xaxis_title='Datum',
        yaxis_title='Aktienkurs',
        template='plotly_white',
        hovermode='x unified'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    fig.show()