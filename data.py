
import os
import tensorflow_datasets as tfds
from utils import EasyDict


# Set default data directory if environment variable is missing
DATA_DIR = os.path.join(os.getenv('ML_DATA', './data'), 'TFDS')

def get_data(data_name: str):
    """
    Load dataset by name and return the training and testing sets, along with
    the number of training samples and number of classes.

    Args:
    - data_name (str): Name of the dataset. Must be one of ['mnist', 'emnist_merge', 'cifar10'].

    Returns:
    - train (EasyDict): Training set containing images and labels.
    - test (EasyDict): Testing set containing images and labels.
    - ntrain (int): Number of training samples.
    - nclass (int): Number of classes.
    """
    valid_datasets = ['mnist', 'emnist_merge', 'cifar10']
    if data_name not in valid_datasets:
        raise ValueError(f"Invalid dataset name '{data_name}'. Must be one of {valid_datasets}.")

    # Adjust dataset name for 'emnist_merge'
    if data_name.startswith('emnist'):
        data_name = 'emnist/by' + data_name[7:]

    # Load dataset with info
    data, info = tfds.load(name=data_name, data_dir=DATA_DIR, batch_size=-1, with_info=True)
    
    # Convert TensorFlow dataset to numpy arrays
    data = tfds.as_numpy(data)

    # Normalize and reshape data
    train_images = data['train']['image'].transpose(0, 3, 1, 2) / 127.5 - 1
    test_images = data['test']['image'].transpose(0, 3, 1, 2) / 127.5 - 1

    # Wrap in EasyDict for cleaner access
    train = EasyDict(image=train_images, label=data['train']['label'])
    test = EasyDict(image=test_images, label=data['test']['label'])

    # Number of training samples and classes
    ntrain = train.image.shape[0]
    nclass = info.features['label'].num_classes

    return train, test, ntrain, nclass
