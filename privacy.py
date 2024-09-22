from absl import app
import numpy as np
import json
from collections import Counter
from typing import List

def convert_gaussian_renyi_to_dp(sigma, delta, verbose=True):
    alphas = np.arange(1, 200, 0.1)[1:]
    epss = alphas / (2 * sigma**2) - (np.log(delta * (alphas - 1)) - alphas * np.log(1 - 1/alphas)) / (alphas - 1)
    idx = np.nanargmin(epss)
    if verbose and idx == len(alphas) - 1:
        print('Increase the range of alpha.')
    return epss[idx]


def get_total_sensitivity_sq(steps_per_epoch, epochs, extra_steps, mem_fn=None):
    if mem_fn:
        mem = json.load(open(mem_fn))
        key = f'{steps_per_epoch},{epochs},{extra_steps}'
        key_no_extra = f'{steps_per_epoch},{epochs},{0}'
        if key in mem and key_no_extra in mem:
            return mem[key], mem[key_no_extra], None
    else:
        mem = {}

    layer = [Counter({ss: 1}) for _ in range(epochs) for ss in range(steps_per_epoch)]
    layer += [Counter({-1: 1}) for _ in range(extra_steps)]
    sensitivity_sq, sensitivity_sq_no_extra = [0] * steps_per_epoch, [0] * steps_per_epoch

    def update_sensitivity_sq(layer):
        for node in layer:
            if -1 in node: has_extra = True
            else: has_extra = False
            for ss in node:
                if ss != -1:
                    sensitivity_sq[ss] += node[ss]**2
                    if not has_extra:
                        sensitivity_sq_no_extra[ss] += node[ss]**2

    update_sensitivity_sq(layer)
    while len(layer) > 1:
        layer = [layer[i] + layer[i + 1] for i in range(0, len(layer) - 1, 2)]
        update_sensitivity_sq(layer)

    if mem_fn:
        mem[key], mem[key_no_extra] = max(sensitivity_sq), max(sensitivity_sq_no_extra)
        json.dump(mem, open(mem_fn, 'w'), indent=4)
    return max(sensitivity_sq), max(sensitivity_sq_no_extra), sensitivity_sq


def compute_epsilon_tree(num_batches: int, epochs_between_restarts: List[int], noise: float, delta: float,
                         tree_completion: bool, mem_fn=None):
    sensitivity_sq = sum(
        get_total_sensitivity_sq(num_batches, e, 2 ** (num_batches * e - 1).bit_length() - num_batches * e if tree_completion else 0, mem_fn)[0]
        for e in epochs_between_restarts if e > 0
    )
    effective_sigma = noise / np.sqrt(sensitivity_sq)
    return convert_gaussian_renyi_to_dp(effective_sigma, delta)


def main(_):
    n, delta, batch, epochs, restart_every, noise = 50000, 1e-5, 500, 100, 20, 46.3
    eps = compute_epsilon_tree(n // batch, [restart_every] * (epochs // restart_every), noise, delta, True)
    print(f'n={n}, batch={batch}, epochs={epochs}, restart={restart_every}, noise={noise}, DP=({eps:.2f}, {delta})')


if __name__ == '__main__':
    app.run(main)
