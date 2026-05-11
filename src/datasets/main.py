from .malaria import Malaria_Dataset
from .nanofibre import Nanofibre_Dataset


def load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0,
                 random_state=None, length=5000, category: str = 'carpet'):
    """Loads the dataset (malaria patches or nanofibre image folders)."""

    implemented_datasets = ('malaria_dataset', 'nanofibre')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'malaria_dataset':
        dataset = Malaria_Dataset(root=data_path,
                                  normal_class=normal_class,
                                  known_outlier_class=known_outlier_class,
                                  n_known_outlier_classes=n_known_outlier_classes,
                                  ratio_known_normal=ratio_known_normal,
                                  ratio_known_outlier=ratio_known_outlier,
                                  ratio_pollution=ratio_pollution)

    if dataset_name == 'nanofibre':
        dataset = Nanofibre_Dataset(root=data_path,
                                    normal_class=normal_class,
                                    known_outlier_class=known_outlier_class,
                                    n_known_outlier_classes=n_known_outlier_classes,
                                    ratio_known_normal=ratio_known_normal,
                                    ratio_known_outlier=ratio_known_outlier,
                                    ratio_pollution=ratio_pollution)

    return dataset
