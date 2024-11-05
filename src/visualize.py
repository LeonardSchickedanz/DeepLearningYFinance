import matplotlib.pyplot as plt


#import data_xlsx as d


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

