from typing import List


class MaximumRetryError(Exception):
    def __init__(self, excs: List[Exception]):
        self.excs = excs

    def __str__(self):
        return str([str(e) for e in self.excs])


class CustomError(ValueError):
    ...
