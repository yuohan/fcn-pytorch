import numpy as np

class Metric:

    def __init__(self, metrics, num_classes):

        self.num_classes = num_classes
        self.cm = np.zeros((num_classes, num_classes)) #confusion metrix
        self.metrics = metrics

    def update(self, y_pred, y_true):

        y_pred = np.argmax(y_pred, axis=1).flatten()
        y_true = y_true.flatten()

        target_mask = (y_true >= 0) & (y_true < self.num_classes)
        y_pred = y_pred[target_mask]
        y_true = y_true[target_mask]

        indices = self.num_classes * y_true + y_pred
        self.cm += np.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        
    def compute(self):

        ESP = 1e-15
        metric_dict = {}
        for m in self.metrics:
            if m == 'accuracy':
                acc = np.diag(self.cm).sum() / (self.cm.sum() + ESP)
                metric_dict[m] = acc
            if m == 'mean_iou':
                iou = np.diag(self.cm) / (self.cm.sum(axis=1) + self.cm.sum(axis=0) - np.diag(self.cm) + ESP)
                metric_dict[m] = iou.mean()
        return metric_dict