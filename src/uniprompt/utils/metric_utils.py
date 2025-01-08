from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np

MetricType = Literal["accuracy", "weighted_accuracy", "hinge_accuracy"]

def get_confusion_matrix(y_true: Sequence[Any], y_pred: Sequence[Any], normalize: bool = False) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    classes.sort()
    n_classes = len(classes)

    cm = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            cm[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))

    if normalize is True:
        for i in range(n_classes):
            if np.sum(cm[i, :]) == 0:
                cm[i, :] = 0
            else:
                cm[i, :] = cm[i, :]/np.sum(cm[i, :])

    return cm

def get_metric(y_true: List[str], y_pred: List[str], config: Optional [Dict] = None) -> float:
    cm = get_confusion_matrix(y_true, y_pred, normalize=True)

    if(config is None):
        metric_type = "accuracy"
    else:
        metric_type: MetricType = config["metric_kwargs"]["type"]

    if metric_type == "accuracy":
        diagonal = cm.diagonal()
        metric_val =  sum(diagonal) / diagonal.shape[0]

    elif metric_type == "weighted_accuracy":
        weights = config["metric_kwargs"]["weights"]
        diagonal = cm.diagonal()
        metric_val = sum(diagonal * weights) / sum(weights)

    elif metric_type == "hinge_accuracy":
        thresholds = config["metric_kwargs"]["thresholds"]
        weights = config["metric_kwargs"]["weights"]
        diagonal = cm.diagonal()
        diagonal.setflags(write=1)
        for i in range(diagonal.shape[0]):
            if diagonal[i] < thresholds[i]:
                diagonal[i] = diagonal[i] / thresholds[i] * diagonal[i]
        metric_val = sum(diagonal * weights) / sum(weights)

    else:
        raise ValueError(f"Metric {metric_type} not supported")

    return metric_val
