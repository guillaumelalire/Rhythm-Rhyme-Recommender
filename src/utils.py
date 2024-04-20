def normalize(arr, min_is_zero=True):
    if min_is_zero:
        return arr / max(arr)
    else:
        return (arr - min(arr)) / (max(arr) - min(arr))