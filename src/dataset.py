from torch.utils.data import Dataset
import random


class TripletMNISTDataset(Dataset):

    def __init__(self, dataset: Dataset, num_samples: int = 60000):
        """
        :param dataset: MNIST dataset from which we want to make a Triplet dataset
        :param num_samples: number of triplets we can
        """
        self.dataset = dataset
        self.num_samples = num_samples
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.images = []
        self.labels2images = {class_name: [] for class_name in self.classes}

        for x, y in self.dataset:
            y = int(y)
            self.labels2images[y].append(x)
            self.images.append((x, y))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        x, y = self.images[item % len(self.images)]

        positive_x = random.choice(self.labels2images[y])

        negative_class = random.choice([class_num for class_num in range(10) if class_num != y])
        negative_x = random.choice(self.labels2images[negative_class])

        return x, positive_x, negative_x
