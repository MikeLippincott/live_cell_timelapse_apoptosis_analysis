class Dataset_formatter:
    """
    A class for formatting data for a data loader

    Attributes:
    ----------
    X : Pandas DataFrame
        the X dimension of data (features)
    Y : Pandas DataFrame
        the Y dimension of data (predictor)

    Methods
    -------
    __len__:
        returns the length of the X dimension
    -------
    __getitem__:
        returns a row of the X and Y dimension given an index
    """

    def __init__(
        self,
        X,
        Y,
    ):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
