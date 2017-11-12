import matplotlib.pyplot as plt


decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plot_node(text, center_plot, parent_plot, node_type):
    create_plot().ax1.annotate(text, xy=parent_plot,
        xycoords='axes fraction', xytext=center_plot,
        textcoords='axes fraction',
        va = "center", ha="center", bbox=node_type,
        arrowprops=arrow_args)

def create_plot():
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    reatePlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('ye',(0.2, 0.1),(0.1, 0.5), decision_node)
    plotNode('jue',(0.8, 0.1),(0.3, 0.8), decision_node)
    plt.show()
