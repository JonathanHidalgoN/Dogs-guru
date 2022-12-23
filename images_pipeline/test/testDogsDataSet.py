import pytest
from images_pipeline.DogsDataSet import DogsDataSet
from images_pipeline.Transformations import Rescale, RandomCrop
from torchvision import transforms
from torch import equal

class TestDogsDataSet:

    def setup_method(self):
        path = "images/Images"
        self.dogsDataSet = DogsDataSet(path)
        
    def test_get_dog_breeds(self):  
        """
        Tests the get_dog_breeds method.
        """
        assert len(self.dogsDataSet.species) == 120
    
    def test_get_full_paths(self):
        """
        Tests the get_full_paths method.
        """
        assert len(self.dogsDataSet._full_paths) == 20580

    def test_add_and_remove(self):
        #I know this is not a good idea to test two methods in one test 
        # but I wanted to test the add and remove methods together because they are related
        #and affect the same attribute
        """
        Tests the add_species method.
        """
        new_species = ["test1", "test2"]
        images_per_specie = [0, 0]
        self.dogsDataSet.add_species(new_species, images_per_specie)
        assert len(self.dogsDataSet.species) == 120 + 2
        self.dogsDataSet.remove_species(new_species)
        assert len(self.dogsDataSet.species) == 120

    def test_len(self):
        """
        Tests the __len__ method.
        """
        assert len(self.dogsDataSet) == 20580

    def test_getitem(self):
        """
        Tests the __getitem__ method.
        """
        transformation = Rescale((256,256))
        self.dogsDataSet.transform = transformation
        assert self.dogsDataSet[0].shape == (3, 256, 256)
        assert self.dogsDataSet[20579].shape == (3, 256, 256)
        self.dogsDataSet.transform = None
        assert self.dogsDataSet[0].shape == (3, 360, 290)
        with pytest.raises(IndexError):
            self.dogsDataSet[20580]
        another_transformation = transforms.Compose([Rescale((256,256)), RandomCrop((224,224))])
        self.dogsDataSet.transform = another_transformation
        assert self.dogsDataSet[0].shape == (3, 224, 224)
        assert self.dogsDataSet[20579].shape == (3, 224, 224)
        self.dogsDataSet.transform = None
        assert len(self.dogsDataSet[[1,2]]) == 2
        two_items = self.dogsDataSet[[0,10]]
        assert type(two_items[0]) == type(two_items[1])
        assert type(two_items[0]) == type(self.dogsDataSet[0])
        assert equal(two_items[0], self.dogsDataSet[0])
        with pytest.raises(TypeError):
            self.dogsDataSet["test"]
        with pytest.raises(IndexError):
            self.dogsDataSet[[1, 20580]]

    

if __name__ == '__main__':
    import subprocess
    subprocess.call(["pytest",str(__file__)])