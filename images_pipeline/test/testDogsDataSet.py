import pytest
from images_pipeline.DogsDataSet import DogsDataSet
from images_pipeline.Transformations import Rescale, RandomCrop, ToTensor
from torchvision import transforms
from torch import equal
from images_pipeline.utils import generate_indexes, count_total_images


class TestDogsDataSet:

    def setup_method(self):
        path = "images/Images"
        self.total_images = count_total_images(path)
        #This is the proportion of the dataset that will be used for training, validation and testing
        self.proportion = [0.8, 0.1, 0.1]
        index_generator = generate_indexes(self.total_images, self.proportion)
        self.datasets = [DogsDataSet(path, index_generator) for _ in self.proportion]

    def test_get_dog_breeds(self):
        """
        Tests the get_dog_breeds method.
        """
        assert len(self.datasets[0].species) == 120
        assert len(self.datasets[1].species) == 120
        assert len(self.datasets[2].species) == 120

    def test_get_full_paths(self):
        """
        Tests the get_full_paths method.
        """
        lens = [len(dataset._full_paths) for dataset in self.datasets]
        assert lens[0] == int(self.total_images * self.proportion[0])
        assert lens[1] == int(self.total_images * self.proportion[1])
        assert lens[2] == int(self.total_images * self.proportion[2])

    def test_number_species(self):
        """
        Tests the number_species method.
        """
        for dataset in self.datasets:
            assert len(dataset.species) == 120

    @pytest.mark.skip(
        reason="This test is not working but now I dont use this method, maybe I will use it in the future"
    )
    def test_add_and_remove(self):
        # I know this is not a good idea to test two methods in one test
        # but I wanted to test the add and remove methods together because they are related
        # and affect the same attribute
        """
        Tests the add_species method.
        """
        new_species = ["test1", "test2"]
        images_per_specie = [0, 0]
        for dataset in self.datasets:
            dataset.add_species(new_species, images_per_specie)
            assert len(dataset.species) == 120 + 2
            dataset.remove_species(new_species)
            assert len(dataset.species) == 120

    def test_len(self):
        """
        Tests the __len__ method.
        """
        lens = [len(dataset) for dataset in self.datasets]
        assert lens[0] == int(self.total_images * self.proportion[0])
        assert lens[1] == int(self.total_images * self.proportion[1])
        assert lens[2] == int(self.total_images * self.proportion[2])

    def test_getitem(self):
        """
        Tests the __getitem__ method.
        """
        # Test the transformations
        transformations = [
            transforms.Compose([Rescale((256, 256)), ToTensor()]),
            transforms.Compose(
                [Rescale((256, 256)), RandomCrop((224, 224)), ToTensor()]
            ),
            transforms.Compose([Rescale((256, 256)), ToTensor()])
        ]
        for idx, dataset in enumerate(self.datasets):
            dataset.transform = transformations[idx]
        assert self.datasets[0][0][0].shape == (3, 256, 256)
        assert self.datasets[0][100][0].shape == (3, 256, 256)
        assert self.datasets[1][0][0].shape == (3, 224, 224)
        assert self.datasets[1][78][0].shape == (3, 224, 224)
        # -------------------------------------------------------------------------
        # Test the index error
        with pytest.raises(IndexError):
            self.datasets[0][20580]
            self.datasets[1][20580]
            self.datasets[2][20580]

        # --------------------------------------------------------------------------
        # Test len of indexing with a list
        assert len(self.datasets[0][[1, 2]][0]) == 2
        assert len(self.datasets[1][[1, 2, 3, 4, 5]][0]) == 5
        assert len(self.datasets[2][[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]][0]) == 10
        # --------------------------------------------------------------------------
        # Test the type of the elements in the list and equality after indexing
        two_items_per_dataset = [
            self.datasets[0][[0, 10]][0] ,
            self.datasets[1][[0, 10]][0],
            self.datasets[2][[0, 10]][0],
        ]
        assert type(two_items_per_dataset[0][0]) == type(two_items_per_dataset[0][1][0])
        assert type(two_items_per_dataset[0][0]) == type(self.datasets[0][0][0])
        assert type(two_items_per_dataset[0][0]) == type(self.datasets[1][0][0])
        assert type(two_items_per_dataset[0][0]) == type(self.datasets[2][0][0])
        # Here we test the equality of the elements in the list and the elements in the dataset
        # Not testing with the second dataset because it has a random crop transformation,
        # so the elements in the list will be different from the elements in the dataset
        assert equal(self.datasets[0][0][0], two_items_per_dataset[0][0])
        assert equal(self.datasets[0][10][0], two_items_per_dataset[0][1])

        assert equal(self.datasets[2][0][0], two_items_per_dataset[2][0])
        assert equal(self.datasets[2][10][0], two_items_per_dataset[2][1])
        assert len(two_items_per_dataset[0]) == 2
        assert len(two_items_per_dataset[1]) == 2
        assert len(two_items_per_dataset[2]) == 2
        # --------------------------------------------------------------------------
        # Test labels
        # Since labels are one hot encoded, the shape of the labels should be equal to the number of species
        assert self.datasets[0][0][1].shape == (len(dataset.species),)
        assert self.datasets[0][100][1].shape == (len(dataset.species),)
        assert self.datasets[1][0][1].shape == (len(dataset.species),)
        del two_items_per_dataset
        # --------------------------------------------------------------------------
        # Test the type of the elements in the list and equality after indexing
        two_labels = [ self.datasets[0][[0, 10]][1], 
                       self.datasets[2][[0, 10]][1],
                       self.datasets[1][[0, 10]][1]]
        assert type(two_labels[0][0]) == type(two_labels[0][1])
        assert type(two_labels[0][0]) == type(self.datasets[0][0][1])
        assert type(two_labels[0][0]) == type(self.datasets[1][0][1])
        assert equal(two_labels[0][0], self.datasets[0][0][1])
        assert equal(two_labels[0][1], self.datasets[0][10][1])
        assert equal(two_labels[1][0], self.datasets[2][0][1])
        assert equal(two_labels[1][1], self.datasets[2][10][1])
        assert equal(two_labels[2][0], self.datasets[1][0][1])
        assert equal(two_labels[2][1], self.datasets[1][10][1])
        assert len(two_labels[0]) == 2
        assert len(two_labels[1]) == 2
        assert len(two_labels[2]) == 2



if __name__ == "__main__":
    import subprocess
    times_to_run = 1
    for i in range(times_to_run):
        subprocess.call(["pytest", str(__file__)])
