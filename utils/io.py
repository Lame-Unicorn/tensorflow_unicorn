import tensorflow as tf
from tqdm import tqdm


def _text_line_generator(filepath, encoding, strip):
    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            if strip:
                line = line.strip()
            else:
                line = line.rstrip("\n")
            if line == "":
                continue
            yield line


def text_line_generator(filepath, encoding="utf-8", strip=True,
                        limit=None, verbose=0):
    generator = _text_line_generator(filepath, encoding, strip)
    if verbose > 0:
        generator = tqdm(generator)

    count = 0
    for x in generator:
        if limit is not None and count >= limit:
            break
        yield x
        count += 1
