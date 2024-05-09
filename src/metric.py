from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        tp = torch.logical_and(preds == target, preds == 1).sum()
        fp = torch.logical_and(preds != target, preds == 1).sum()
        fn = torch.logical_and(preds != target, target == 1).sum()

        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 / (1/precision + 1/recall + 1e-6) # 1e-6 prevent division by zero
        return f1
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError("Predictions and targets have different shapes.")

        # [TODO] Count the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
