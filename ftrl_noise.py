import torch
from collections import namedtuple
from absl import app

class CummuNoiseTorch:
    @torch.no_grad()
    def __init__(self, std: float, shapes: list, device: str, test_mode: bool = False):
        """
        :param std: standard deviation of the noise
        :param shapes: shapes of the noise, which is basically shape of the gradients
        :param device: device for pytorch tensor
        :param test_mode: if in test mode, noise will be 1 in each node of the tree
        """
        assert std >= 0, "Standard deviation must be non-negative"
        self.std = std
        self.shapes = shapes
        self.device = device
        self.step = 0
        self.binary = [0]
        self.noise_sum = [torch.zeros(shape, device=device) for shape in shapes]
        self.recorded = [[torch.zeros(shape, device=device) for shape in shapes]]
        self.test_mode = test_mode

    @torch.no_grad()
    def __call__(self) -> list:
        """
        :return: the noise to be added by DP-GFTPRL
        """
        if self.std <= 0 and not self.test_mode:
            return self.noise_sum

        self.step += 1

        idx = 0
        while idx < len(self.binary) and self.binary[idx] == 1:
            self.binary[idx] = 0
            for ns, re in zip(self.noise_sum, self.recorded[idx]):
                ns -= re
            idx += 1

        if idx >= len(self.binary):
            self.binary.append(0)
            self.recorded.append([torch.zeros(shape, device=self.device) for shape in self.shapes])

        for ns, re in zip(self.noise_sum, self.recorded[idx]):
            n = torch.ones_like(ns) if self.test_mode else torch.normal(0, self.std, ns.shape, device=self.device)
            ns.add_(n)
            re.copy_(n)

        self.binary[idx] = 1
        return self.noise_sum

    @torch.no_grad()
    def proceed_until(self, step_target: int) -> list:
        """
        Proceed until the step_target-th step.

        :return: the noise to be added by DPGFTPRL
        """
        if self.step >= step_target:
            raise ValueError(f'Already reached {step_target}.')
        while self.step < step_target:
            noise_sum = self.__call__()
        return noise_sum


Element = namedtuple('Element', 'height value')


class CummuNoiseEffTorch:
    """
    The tree aggregation protocol with the trick in Honaker, "Efficient Use of Differentially Private Binary Trees", 2015
    """
    @torch.no_grad()
    def __init__(self, std: float, shapes: list, device: str):
        """
        :param std: standard deviation of the noise
        :param shapes: shapes of the noise, which is basically shape of the gradients
        :param device: device for pytorch tensor
        """
        self.std = std
        self.shapes = shapes
        self.device = device
        self.step = 0
        self.noise_sum = [torch.zeros(shape, device=device) for shape in shapes]
        self.stack = []

    @torch.no_grad()
    def get_noise(self) -> list:
        """Generate noise based on the provided standard deviation and shapes."""
        return [torch.normal(0, self.std, shape, device=self.device) for shape in self.shapes]

    @torch.no_grad()
    def push(self, elem: Element):
        """Push a new element onto the stack and update the noise sum."""
        factor = 2.0 - 1 / 2 ** elem.height
        for i, shape in enumerate(self.shapes):
            self.noise_sum[i] += elem.value[i] / factor
        self.stack.append(elem)

    @torch.no_grad()
    def pop(self):
        """Pop the top element from the stack and update the noise sum."""
        elem = self.stack.pop()
        factor = 2.0 - 1 / 2 ** elem.height
        for i, shape in enumerate(self.shapes):
            self.noise_sum[i] -= elem.value[i] / factor

    @torch.no_grad()
    def __call__(self) -> list:
        """
        :return: the noise to be added by DP-GFTPRL
        """
        self.step += 1

        # Add new element to the stack
        self.push(Element(0, self.get_noise()))

        # Pop and combine elements with the same height
        while len(self.stack) >= 2 and self.stack[-1].height == self.stack[-2].height:
            left_value, right_value = self.stack[-2].value, self.stack[-1].value
            new_noise = self.get_noise()
            new_elem = Element(
                self.stack[-1].height + 1,
                [x + (y + z) / 2 for x, y, z in zip(new_noise, left_value, right_value)]
            )

            # Pop twice and push the new element
            self.pop()
            self.pop()
            self.push(new_elem)

        return self.noise_sum

    @torch.no_grad()
    def proceed_until(self, step_target: int) -> list:
        """
        Proceed until the step_target-th step.

        :return: the noise to be added by DP-GFTPRL
        """
        if self.step >= step_target:
            raise ValueError(f'Already reached {step_target}.')
        while self.step < step_target:
            noise_sum = self.__call__()
        return noise_sum


def main(argv):
    """Test function for the CummuNoiseTorch class."""
    def count_set_bits(n: int) -> int:
        """Count the number of set bits (1s) in the binary representation of a number."""
        return bin(n).count('1')

    cummu_noises = CummuNoiseTorch(1.0, [(1,)], 'cuda', test_mode=True)
    for epoch in range(31):
        random_noise = cummu_noises()
        assert random_noise[0].cpu().numpy()[0] == count_set_bits(epoch + 1)


if __name__ == '__main__':
    app.run(main)
