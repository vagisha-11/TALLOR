from typing import Union
import torch
import numpy as np
from collections import defaultdict

ArrayLike = Union[torch.Tensor, np.ndarray, list]

def to_tensor_long(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.long, device='cpu')
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=torch.long, device='cpu')
    return torch.as_tensor(x, dtype=torch.long, device='cpu')

def to_tensor_float(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32, device='cpu')
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=torch.float32, device='cpu')
    return torch.as_tensor(x, dtype=torch.float32, device='cpu')


class PrecisionRecallF1:
    """
    compute precision recall and f1
    """
    def __init__(self, neg_label: int, reduce: str = 'micro', binary_match: bool = False) -> None:
        self._neg_label = neg_label
        assert reduce in {'micro', 'macro'}
        if reduce == 'macro':
            raise Exception("precision, recall, and F1 don't have macro version?")
        self._reduce = reduce
        self._binary_match = binary_match  # only consider the existence
        self._count = 0.0
        self._precision = 0.0
        self._recall = 0.0
        self._bucket = {
            'count': defaultdict(lambda: 0.0),
            'precision': defaultdict(lambda: 0.0),
            'recall': defaultdict(lambda: 0.0),
        }
        self._recall_local = 0.0
        self._f1 = 0.0
        self._num_sample = 0
        self._used = False

    def __call__(
        self,
        predictions: ArrayLike,  # (batch_size, seq_len)
        labels: ArrayLike,       # (batch_size, seq_len)
        mask: ArrayLike,         # (batch_size, seq_len)
        recall: ArrayLike = None,
        duplicate_check: bool = True,
        bucket_value: ArrayLike = None,  # (batch_size, seq_len)
    ):
        # Normalize inputs (supports list/np/tensor)
        preds_t = to_tensor_long(predictions)
        labels_t = to_tensor_long(labels)
        mask_t = to_tensor_long(mask)

        if preds_t.dim() != 2:
            raise Exception('inputs should have two dimensions')
        self._used = True

        # Build predicted/true masks as float for accumulation
        predicted = (preds_t.ne(self._neg_label).long() * mask_t).float()
        whole_subset = (labels_t.ne(self._neg_label).long() * mask_t).float()
        self._recall_local += float(whole_subset.sum().item())

        if recall is not None:
            whole_t = to_tensor_float(recall)
            if duplicate_check:
                assert whole_subset.sum().item() <= whole_t.sum().item(), 'found duplicate span pairs'
        else:
            whole_t = whole_subset

        if self._binary_match:
            matched = (preds_t.ne(self._neg_label).long() * labels_t.ne(self._neg_label).long() * mask_t).float()
        else:
            matched = (preds_t.eq(labels_t).long() * mask_t * preds_t.ne(self._neg_label).long()).float()

        if self._reduce == 'micro':
            self._count += float(matched.sum().item())
            self._precision += float(predicted.sum().item())
            self._recall += float(whole_t.sum().item())
            self._num_sample += preds_t.size(0)

            if bucket_value is not None:
                # Prepare arrays for np.bincount (int indices, float weights)
                bucket_np = (to_tensor_long(bucket_value) * mask_t).cpu().numpy().reshape(-1).astype(np.int64)
                matched_np = matched.cpu().numpy().reshape(-1)
                predicted_np = predicted.cpu().numpy().reshape(-1)
                whole_subset_np = whole_subset.cpu().numpy().reshape(-1)

                count = np.bincount(bucket_np, weights=matched_np)
                precision = np.bincount(bucket_np, weights=predicted_np)
                recall_arr = np.bincount(bucket_np, weights=whole_subset_np)

                for b, v in enumerate(count):
                    self._bucket['count'][b] += float(v)
                for b, v in enumerate(precision):
                    self._bucket['precision'][b] += float(v)
                for b, v in enumerate(recall_arr):
                    self._bucket['recall'][b] += float(v)

        else:
            # Kept for completeness though 'macro' is disallowed by constructor
            self._count += float(matched.numel())
            pre = matched / (predicted + 1e-10)
            rec = matched / (whole_t + 1e-10)
            f1 = 2 * pre * rec / (pre + rec + 1e-10)
            self._precision += float(pre.sum().item())
            self._recall += float(rec.sum().item())
            self._f1 += float(f1.sum().item())
            self._num_sample += preds_t.size(0)

    def get_metric(self, reset: bool = False):
        if not self._used:
            return None
        if self._reduce == 'micro':
            p = self._count / (self._precision + 1e-10)
            r = self._count / (self._recall + 1e-10)
            f = 2 * p * r / (p + r + 1e-10)
            included_recall = self._recall_local / (self._recall + 1e-10)
            m = {'p': p, 'r': r, 'f': f, 'r_': included_recall}
            if len(self._bucket['count']) > 0:
                for b in list(self._bucket['count'].keys()):
                    bp = self._bucket['count'][b] / (self._bucket['precision'][b] + 1e-10)
                    br = self._bucket['count'][b] / (self._bucket['recall'][b] + 1e-10)
                    bf = 2 * bp * br / (bp + br + 1e-10)
                    m[f'bucket_f_{b}'] = bf
                for b in list(self._bucket['count'].keys()):
                    m[f'bucket_all_{b}'] = self._bucket['recall'][b]
        else:
            m = {
                'p': self._precision / (self._count + 1e-10),
                'r': self._recall / (self._count + 1e-10),
                'f': self._f1 / (self._count + 1e-10),
            }
        if reset:
            self.reset()
        return m

    def reset(self):
        self._count = 0.0
        self._precision = 0.0
        self._recall = 0.0
        self._bucket = {
            'count': defaultdict(lambda: 0.0),
            'precision': defaultdict(lambda: 0.0),
            'recall': defaultdict(lambda: 0.0),
        }
        self._recall_local = 0.0
        self._f1 = 0.0
        self._num_sample = 0
        self._used = False

    def detach_tensors(self, *tensors):
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)


class DataPrecisionRecallF1:
    """
    compute precision recall and f1 for datapoint
    """
    def __init__(self, neg_label: int) -> None:
        self._neg_label = neg_label
        self._count = 0.0
        self._precision = 0.0
        self._recall = 0.0
        self._recall_local = 0.0
        self._f1 = 0.0
        self._used = False

    def __call__(
        self,
        predictions: ArrayLike,  # (seq_len)
        labels: ArrayLike,       # (seq_len)
        mask: ArrayLike,         # (seq_len)
    ):
        # Normalize to CPU tensors first
        preds_t = to_tensor_long(predictions)
        labels_t = to_tensor_long(labels)
        mask_f = to_tensor_float(mask)

        self._used = True
        predicted = ((preds_t != self._neg_label).long().float() * mask_f).double()
        whole_subset = (labels_t != self._neg_label).long().double()
        self._recall_local += float(whole_subset.sum().item())

        matched = ((preds_t == labels_t).long() * (preds_t != self._neg_label).long()).float() * mask_f
        matched = matched.double()

        self._count += float(matched.sum().item())
        self._precision += float(predicted.sum().item())
        self._recall += float(whole_subset.sum().item())

    def get_metric(self, reset: bool = False):
        if not self._used:
            return None
        p = self._count / (self._precision + 1e-10)
        r = self._count / (self._recall + 1e-10)
        f = 2 * p * r / (p + r + 1e-10)
        included_recall = self._recall_local / (self._recall + 1e-10)
        m = {'p': p, 'r': r, 'f': f, 'r_': included_recall}
        if reset:
            self.reset()
        return m

    def reset(self):
        self._count = 0.0
        self._precision = 0.0
        self._recall = 0.0
        self._recall_local = 0.0
        self._f1 = 0.0
        self._used = False
