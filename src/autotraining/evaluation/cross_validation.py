# src/autotraining/evaluation/cross_validation.py

def make_walk_forward_folds(
    total_length: int,
    min_train_size: int,
    horizon: int,
    stride: int,
):

    folds = []

    train_end = min_train_size

    while train_end + horizon <= total_length:

        folds.append(
            (train_end, train_end + horizon)
        )

        train_end += stride

    return folds