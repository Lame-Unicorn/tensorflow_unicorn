from numpy import allclose, array, square, sum
from pandas import cut, qcut


def normalize(column, mean=None, std=None):
    if mean is None:
        mean = column.mean()

    if std is None:
        std = column.std()
    return (column - mean) / std, (mean, std)


def discretize(column, /, n_bins=None, bins=None):
    if n_bins is not None:
        return qcut(column, q=n_bins, labels=False, retbins=True, duplicates="drop")
    elif bins is not None:
        return cut(column, bins=bins, labels=False, include_lowest=True)
    else:
        raise Warning("Neither n_bins or bins is settled.")


def stats(sample_x, sample_y, cumsums=None):
    if cumsums == None:
        cumsums = [[0, 0], [0, 0], 0]

    cumsums[0][0] += sum(sample_x, axis=0)
    cumsums[0][1] += sum(square(sample_x), axis=0)
    cumsums[1][0] += sum(sample_y, axis=0)
    cumsums[1][1] += sum(square(sample_y), axis=0)
    cumsums[2] += sample_x.shape[0]

    return (
        (cumsums[0][0] / cumsums[2], cumsums[0][1] / cumsums[2] - (cumsums[0][0] / cumsums[2]) ** 2)
    )


if __name__ == "__main__":
    x = array([1, 2, 3, 3, 4, 5, 6, 6, 6, 7, 8, 8])
    print(normalize(x))
    c1, bins = discretize(x, n_bins=8)
    c2 = discretize(x, bins=bins)
    print(allclose(c1, c2))
