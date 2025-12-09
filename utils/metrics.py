import torch

def iou_score(y_pred, y_true, threshold=0.5):
    """Tính Intersection over Union (IoU)."""
    y_pred = (y_pred > threshold).float().view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def f1_score(y_pred, y_true, threshold=0.5):
    """Tính F1-score."""
    y_pred = (y_pred > threshold).float()
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return 2 * (precision * recall) / (precision + recall + 1e-6)