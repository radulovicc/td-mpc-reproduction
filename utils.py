import re
import numpy as np
import torch
import torch.nn as nn

def linear_schedule(schdl, step):
    """
    Linear layout for values like std or horizon.
    PReceives a string in form of "linear(init,final,duration)".
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)

def ema(model, model_target, tau):
    """Updates the target network with an exponential moving average."""
    with torch.no_grad():
        for p, p_target in zip(model.parameters(), model_target.parameters()):
            p_target.data.lerp_(p.data, tau)