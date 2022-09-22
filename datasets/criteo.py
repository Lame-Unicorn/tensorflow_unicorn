"""This dataset is released along with the paper: “A Large Scale Benchmark for Uplift Modeling” Eustache Diemert,
Artem Betlei, Christophe Renaudin; (Criteo AI Lab), Massih-Reza Amini (LIG, Grenoble INP)

This work was published in: AdKDD 2018 Workshop, in conjunction with KDD 2018.

Data description This dataset is constructed by assembling data resulting from several incrementality tests,
a particular randomized trial procedure where a random part of the population is prevented from being targeted by
advertising. it consists of 25M rows, each one representing a user with 11 features, a treatment indicator and 2
labels (visits and conversions).

Fields
Here is a detailed description of the fields (they are comma-separated in the file):
    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11: feature values (dense, float)
    treatment: treatment group (1 = treated, 0 = control)
    conversion: whether a conversion occured for this user (binary, label)
    visit: whether a visit occured for this user (binary, label)
    exposure: treatment effect, whether the user has been effectively exposed (binary)

"""

from tensorflow import data, stack
from os import path
from numpy import transpose, array


def __load_dataset(filepath, batch_size, features, label_name):
    if not path.exists(filepath):
        raise FileNotFoundError(f"Require manual download file \'{filepath}\' from "
                                "https://ailab.criteo.com/criteo-uplift-prediction-dataset/ !")

    if features is None:
        features = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11'] + [label_name]

    if label_name not in features:
        features.append(label_name)

    dataset = data.experimental.make_csv_dataset(
        filepath,
        batch_size=batch_size,
        select_columns=features, label_name=label_name,
    )

    return dataset


def load_data(
        batch_size, features=None,
        label_name="visit", shuffle=True,
        filepath="tensorflow_unicorn\\datasets_cache\\criteo\\criteo-uplift-v2.1.csv"
):
    criteo_ds = __load_dataset(filepath, batch_size, features=features, label_name=label_name)
    if shuffle:
        criteo_ds = criteo_ds.shuffle(2048)

    def map_fn(x, y):
        x = list(x.values())
        return stack(x, axis=1), y

    criteo_ds = criteo_ds.map(map_fn).prefetch(4)
    return criteo_ds


if __name__ == '__main__':
    dataset = load_data(4, filepath="..\\datasets_cache\\criteo\\criteo-uplift-v2.1.csv")
    for sample in dataset:
        break
    print(sample[0].shape, sample[1].shape)
