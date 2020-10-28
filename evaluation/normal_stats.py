import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def unify_pred_true_y(true_y, pred_y):
    assert isinstance(true_y, np.ndarray)
    assert isinstance(pred_y, np.ndarray)
    if true_y.ndim == 2:
        true_y = np.argmax(true_y, axis=1)
    if pred_y.ndim == 2:
        pred_y = np.argmax(pred_y, axis=1)
    assert true_y.shape == pred_y.shape
    return true_y, pred_y


def confusion(true_y, pred_y, plot_cm=True, labels=None, save_path=None):
    """Calculate confusion matrix.
    Args:
        true_y: groud truth label without one-hot encoding.
        pred_y: predicted label (threshold = max), the same shape as true_y.
        plot_cm: whether or not plot the confusion matrix.
        labels: labels.
    Return:
        A confusion matrix array.
    """
    m = confusion_matrix(true_y, pred_y)
    if plot_cm:
        if labels is None:
            labels = [i for i in range(m.shape[0])]

        tick_marks = np.array(range(len(labels))) + 0.5
        plt.figure(figsize=(len(labels)+2, len(labels)+2), dpi=120)
        plt.imshow(m, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=0, fontsize=8)
        plt.yticks(xlocations, labels, fontsize=8)
        plt.ylabel('Ground truth label')
        plt.xlabel('Predicted label')

        for x_val in range(len(labels)):
            for y_val in range(len(labels)):
                plt.text(x_val, y_val, "{}".format(
                    m[y_val][x_val]), color='red', fontsize=10, va='center', ha='center')

        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid('off')
#         plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig(save_path, format='eps')
    return m


def accuracy(true_y, pred_y):
    return sum(true_y == pred_y) / len(true_y)
