def divide_into_batches(iterable, n=1):
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i:min(i + n, l)]