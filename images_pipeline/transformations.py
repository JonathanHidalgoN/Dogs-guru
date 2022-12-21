from torch import Tensor as torch_tensor
import typing

class Rescale:

    """
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.

    """

    def __init__(self, output_size : typing.Tuple[int, int]):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image : torch_tensor) -> torch_tensor:
        pass
