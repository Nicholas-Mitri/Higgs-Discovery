from HiggsBosonCompetition_AMSMetric_rev1 import AMS
import numpy as np


def AMS_scoring(ground_truth=None, predictions=None, **kwargs):
    w = kwargs['w']
    ground_truth.reshape(-1, 1)
    predictions.reshape(-1, 1)
    s = np.sum(w * ground_truth * predictions)
    ground_truth_transf = np.array([1 if c == 0 else 0 for c in ground_truth])
    b = np.sum(w * ground_truth_transf * predictions)
    return AMS(s, b)


if __name__ == "__main__":
    g = np.array([1, 0, 0, 1])
    p = np.array([1, 0, 1, 1])
    w = np.array([1, 1, 1, 1])

    print(AMS_scoring(g, p, w=w)) # expected answer is 0.586
