import pandas as pd

pd.options.display.max_columns = 50


def load_data(
        path: str,
        index=None,
        drop=None,
        columns=None,
        norm=None,
        oneHot=None
    ):

    # loads data
    df = pd.read_csv(path)

    # sets index
    if index is not None:
        df = df.set_index(index)

    # removes unwanted columns
    if drop is not None:
        for col in drop:
            del df[col]

    # remove all missing values
    for col in df:
        df = df[df[col].notna()]

    # re-names columns
    if columns is not None:
        df = df.rename(columns=columns)

    # normalised to the Z-score
    if norm is not None:
        for col in norm:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    # 1 hot encoding
    if oneHot is not None:
        for col in oneHot:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
            del df[col]

    return df


def match(
        test: pd.DataFrame,
        testY: pd.DataFrame
    ) -> pd.DataFrame:

    for x in testY.index:
        if not x in test.index:
            testY = testY.drop(index=x)

    return testY