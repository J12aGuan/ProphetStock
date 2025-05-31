# graph.py

import matplotlib.pyplot as plt
from stockPrediction.graph.plotData import Plot

plots = []
current = [0]  # mutable index
fig, ax = plt.subplots()

# Build your plot objects (store data, not rendered figures)
def storePlotData(stock, year, w, b):
    plotObj = Plot(stock, year, w, b)
    plots.append(plotObj)

def updatePlot(ax):
    ax.clear()
    plot = plots[current[0]]
    dates, actual, predicted = plot.getPlotData()

    ax.plot(dates, actual, label="Actual Closing Price")
    ax.plot(dates, predicted, label="Predicted Closing Price", linestyle='--')
    ax.set_title(f"{plot.stock} Actal Closing Price vs. Predicted Closing Price (LR)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (USD)")
    ax.legend()
    ax.grid(True)
    fig.canvas.draw()

def onKey(event):
    if event.key == 'right':
        current[0] = (current[0] + 1) % len(plots)
    elif event.key == 'left':
        current[0] = (current[0] - 1) % len(plots)
    updatePlot(ax)

def showGraph():
    fig.canvas.mpl_connect('key_press_event', onKey)

    updatePlot(ax)
    plt.show()
