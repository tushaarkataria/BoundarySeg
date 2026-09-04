"""Ramp-up schedules for consistency weighting.

Clean-room implementation of the sigmoid (Gaussian) ramp-up published as
equation (1)-adjacent text in Laine & Aila, "Temporal Ensembling for
Semi-Supervised Learning", ICLR 2017 (arXiv:1610.02242), which specifies the
weighting function

    w(t) = exp(-5 * (1 - t/T)^2)     for 0 <= t <= T,   w(t) = 1 for t > T

where t is the current step and T the ramp length. Written from the paper's
formula so that this file carries no third-party code.
"""

import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Gaussian ramp-up from 0 to 1 over `rampup_length` steps.

    Returns 1.0 for a zero-length ramp (i.e. no ramping) and saturates at 1.0
    once `current` reaches `rampup_length`.
    """
    if rampup_length == 0:
        return 1.0
    phase = 1.0 - np.clip(current, 0.0, rampup_length) / rampup_length
    return float(np.exp(-5.0 * phase * phase))
