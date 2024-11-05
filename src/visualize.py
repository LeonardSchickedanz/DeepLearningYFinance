import matplotlib.pyplot as plt
import numpy as np
import torch
#import data as d
from src.data import d_quarterly_income
import pandas as pd
from datetime import datetime, timedelta


def plot_losses(losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_stocks(data):
    plt.plot(data)
    plt.title('Test')
    plt.show()
    plt.grid()

