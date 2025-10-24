from typing import List, Union
import torch
import numpy as np
from collections import defaultdict

class PrecisionRecallF1:
    """
    compute precision recall and f1
    """
    def __init__(self, neg_label: int, reduce: str = 'micro', binary_match: bool = False) -> None:
        self._neg_label = neg_label  # negative/ignore label id [web:221][web:224]
        assert reduce in {'micro', 'macro'}  # allowed reductions [web:221]
        if reduce == 'macro':
            raise Exception("precision, recall, and F1 don't have macro version?")
        self._reduce = reduce  # reduction method [web:221]
        self._binary_match = binary_match  # only existence, not class match [web:221]
        self._count = 0  # matched count [web:221]
        self._precision = 0  # predicted positives [web:221]
        self._recall = 0  # true positives + false negatives (denominator) [web:221]
        self._bucket = {  # optional bucketing [web:221]
            'count': defaultdict(lambda: 0),
            'precision': defaultdict(lambda: 0),
            'recall': defaultdict(lambda: 0),
        }
        self._recall_local = 0  # local recall accumulator [web:221]
        self._f1 = 0  # f1 accumulator for macro (unused here) [web:221]
        self._num_sample = 0  # sample count [web:221]
        self._used = False  # flag to indicate __call__ was invoked [web:221]

    def __call__(
        self,
        predictions: torch.LongTensor,  # (batch_size, seq_len) [web:221]
        labels: torch.LongTensor,       # (batch_size, seq_len) [web:221]
        mask: torch.LongTensor,         # (batch_size, seq_len) [web:221]
        recall: torch.LongTensor = None,  # (batch_size, seq_len) [web:221]
        duplicate_check: bool = True,
        bucket_value: torch.LongTensor = None,  # (batch_size, seq_len) [web:221]
    ):
        if len(predictions.size()) != 2:
            raise Exception('inputs should have two dimensions')  # shape guard [web:221]
        self._used = True  # mark as used [web:221]

        # Use int64 instead of deprecated np.long; cast mask to float for arithmetic [web:221][web:224]
        predicted = ((predictions != self._neg_label).long() * mask.long()).float()  # tensor ops avoid numpy casts [web:221][web:224]
        whole_subset = (labels.ne(self._neg_label).long() * mask.long()).float()  # positive labels within mask [web:221][web:224]
        self._recall_local += whole_subset.sum().item()  # accumulate recall denominator observed [web:221]

        if recall is not None:
            whole = recall.float()  # external recall mask [web:221]
            if duplicate_check:
                assert whole_subset.sum().item() <= whole.sum().item(), 'found duplicate span pairs'  # consistency check [web:221]
        else:
            whole = whole_subset  # use computed subset if recall not provided [web:221]

        if self._binary_match:
            matched = (predictions.ne(self._neg_label).long() * labels.ne(self._neg_label).long() * mask.long()).float()  # existence match [web:221]
        else:
            matched = (predictions.eq(labels).long() * mask.long() * predictions.ne(self._neg_label).long()).float()  # class match [web:221]

        if self._reduce == 'micro':
            self._count += matched.sum().item()  # TP [web:221]
            self._precision += predicted.sum().item()  # predicted positives [web:221]
            self._recall += whole.sum().item()  # actual positives [web:221]
            self._num_sample += predictions.size(0)  # batch count [web:221]

            if bucket_value is not None:
                # Move to CPU/NumPy for bincount; ensure integer indices [web:221][web:224]
                bucket_value_np = (bucket_value * mask.long()).cpu().numpy().reshape(-1).astype(np.int64)  # buckets [web:221]
                matched_np = matched.cpu().numpy().reshape(-1)  # matched mask [web:221]
                predicted_np = predicted.cpu().numpy().reshape(-1)  # predicted mask [web:221]
                whole_subset_np = whole_subset.cpu().numpy().reshape(-1)  # true mask [web:221]

                count = np.bincount(bucket_value_np, weights=matched_np)  # per-bucket TP [web:221]
                precision = np.bincount(bucket_value_np, weights=predicted_np)  # per-bucket predicted positives [web:221]
                recall = np.bincount(bucket_value_np, weights=whole_subset_np)  # per-bucket true positives denom [web:221]

                for name in ['count', 'precision', 'recall']:
                    value = {'count': count, 'precision': precision, 'recall': recall}[name]  # select array [web:221]
                    for b, v in enumerate(value):
                        self._bucket[name][b] += float(v)  # accumulate [web:221]

        elif self._reduce == 'macro':
            # Kept for completeness though constructor blocks macro; retains original behavior [web:221]
            self._count += matched.size(0)  # total elements [web:221]
            pre = matched / (predicted + 1e-10)  # per-element precision proxy [web:221]
            rec = matched / (whole + 1e-10)  # per-element recall proxy [web:221]
            f1 = 2 * pre * rec / (pre + rec + 1e-10)  # per-element f1 proxy [web:221]
            self._precision += pre.sum().item()  # sum precision [web:221]
            self._recall += rec.sum().item()  # sum recall [web:221]
            self._f1 += f1.sum().item()  # sum f1 [web:221]
            self._num_sample += predictions.size(0)  # batch count [web:221]

    def get_metric(self, reset: bool = False):
        if not self._used:
            return None  # no calls yet [web:221]
        if self._reduce == 'micro':
            p = self._count / (self._precision + 1e-10)  # precision [web:221]
            r = self._count / (self._recall + 1e-10)  # recall [web:221]
            f = 2 * p * r / (p + r + 1e-10)  # f1 [web:221]
            included_recall = self._recall_local / (self._recall + 1e-10)  # coverage ratio [web:221]
            m = {'p': p, 'r': r, 'f': f, 'r_': included_recall}  # metrics dict [web:221]
            if len(self._bucket['count']) > 0:
                for b, _ in self._bucket['count'].items():
                    bp = self._bucket['count'][b] / (self._bucket['precision'][b] + 1e-10)  # bucket precision [web:221]
                    br = self._bucket['count'][b] / (self._bucket['recall'][b] + 1e-10)  # bucket recall [web:221]
                    bf = 2 * bp * br / (bp + br + 1e-10)  # bucket f1 [web:221]
                    m[f'bucket_f_{b}'] = bf  # store [web:221]
                for b, _ in self._bucket['count'].items():
                    m[f'bucket_all_{b}'] = self._bucket['recall'][b]  # bucket totals [web:221]
        elif self._reduce == 'macro':
            m = {
                'p': self._precision / (self._count + 1e-10),
                'r': self._recall / (self._count + 1e-10),
                'f': self._f1 / (self._count + 1e-10),
            }  # aggregate macro metrics [web:221]
        if reset:
            self.reset()  # clear state [web:221]
        return m  # return metrics [web:221]

    def reset(self):
        self._count = 0  # reset counters [web:221]
        self._precision = 0  # reset precision sum [web:221]
        self._recall = 0  # reset recall sum [web:221]
        self._bucket = {
            'count': defaultdict(lambda: 0),
            'precision': defaultdict(lambda: 0),
            'recall': defaultdict(lambda: 0),
        }  # reset buckets [web:221]
        self._recall_local = 0  # reset local recall [web:221]
        self._f1 = 0  # reset f1 sum [web:221]
        self._num_sample = 0  # reset sample count [web:221]
        self._used = False  # mark unused [web:221]

    def detach_tensors(self, *tensors):
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)  # safe detach [web:221]


class DataPrecisionRecallF1:
    """
    compute precision recall and f1 for datapoint
    """
    def __init__(self, neg_label: int) -> None:
        self._neg_label = neg_label  # ignore label id [web:221]
        self._count = 0  # TP [web:221]
        self._precision = 0  # predicted positives [web:221]
        self._recall = 0  # true positives denom [web:221]
        self._recall_local = 0  # coverage [web:221]
        self._f1 = 0  # not used in micro [web:221]
        self._used = False  # call flag [web:221]

    def __call__(
        self,
        predictions: torch.LongTensor,  # (seq_len) [web:221]
        labels: torch.LongTensor,       # (seq_len) [web:221]
        mask: torch.LongTensor,         # (seq_len) [web:221]
    ):
        # Convert to NumPy with explicit dtypes; avoid deprecated np.long [web:221][web:224]
        predictions = np.asarray(predictions.cpu().numpy(), dtype=np.int64)  # ints [web:221]
        labels = np.asarray(labels.cpu().numpy(), dtype=np.int64)  # ints [web:221]
        mask = np.asarray(mask.cpu().numpy(), dtype=np.float64)  # float mask [web:221]

        self._used = True  # mark used [web:221]
        predicted = ((predictions != self._neg_label).astype(np.int64) * mask).astype(np.float64)  # predicted mask [web:221][web:224]
        whole_subset = (labels != self._neg_label).astype(np.float64)  # true positives denom [web:221]
        self._recall_local += whole_subset.sum()  # accumulate recall denom [web:221]
        whole = whole_subset  # used denom [web:221]

        matched = (
            (predictions == labels).astype(np.int64) * (predictions != self._neg_label).astype(np.int64) * mask
        ).astype(np.float64)  # TP mask [web:221]

        self._count += matched.sum()  # TP [web:221]
        self._precision += predicted.sum()  # predicted positives [web:221]
        self._recall += whole.sum()  # true positives denom [web:221]

    def get_metric(self, reset: bool = False):
        if not self._used:
            return None  # no calls [web:221]
        p = self._count / (self._precision + 1e-10)  # precision [web:221]
        r = self._count / (self._recall + 1e-10)  # recall [web:221]
        f = 2 * p * r / (p + r + 1e-10)  # f1 [web:221]
        included_recall = self._recall_local / (self._recall + 1e-10)  # coverage [web:221]
        m = {'p': p, 'r': r, 'f': f, 'r_': included_recall}  # metrics dict [web:221]
        if reset:
            self.reset()  # reset state [web:221]
        return m  # return metrics [web:221]

    def reset(self):
        self._count = 0  # reset TP [web:221]
        self._precision = 0  # reset precision [web:221]
        self._recall = 0  # reset recall [web:221]
        self._recall_local = 0  # reset coverage [web:221]
        self._f1 = 0  # reset f1 [web:221]
        self._used = False  # clear flag [web:221]
