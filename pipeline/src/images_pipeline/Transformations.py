#Define some classes to be used in the pipeline. These classes will be used to transform the images in the dataset.
#SOURCE:https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from torch import Tensor as torch_tensor
import typing
from torch import randint

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
        """
        Rescale the image in a sample to a given size.
        Args:
            image (Tensor): Image to be scaled.
        Returns:
            Tensor: Rescaled image.
        """
        h, w = image.shape[:2]
        #Check if the output size is a tuple or an integer
        if isinstance(self.output_size, int):
            if h > w:
                #If the height is greater than the width, then the new height is the output size 
                # and the new width is the output size times the width divided by the height
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                #If the width is greater than the height, then the new width is the output size
                new_h, new_w = self.output_size, self.output_size * w / h
        #If the output size is a tuple, then the new height is the first element of the tuple
        #and the new width is the second element of the tuple
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        #Resize the image
        return image.resize((new_h, new_w))



class RandomCrop:

    """
    Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size : typing.Tuple[int, int]):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, image : torch_tensor) -> torch_tensor:
        """
        Crop randomly the image in a sample.
        Args:
            image (Tensor): Image to be cropped.
        Returns:
            Tensor: Cropped image.
        """
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        #In the tutorial they suggest to use pytorch's random function, they use numpy's randint function
        top = randint(0, h - new_h)
        left = randint(0, w - new_w)

        return image[top: top + new_h, left: left + new_w]


if "__name__" == "__main__":
    rescale = Rescale((100, 100))