def normalize(arr, min_is_zero=True):
    """
    Normalizes the given array: its values are scaled to a range between 0 and 1.
    """
    if min_is_zero:
        return arr / max(arr)
    else:
        return (arr - min(arr)) / (max(arr) - min(arr))