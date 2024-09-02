import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Optional
from functools import reduce

# number of clients
K = 300


def read_file(filename: str) -> Optional[List[str]]:
    try:
        with open(filename, "r") as file:
            data = file.read()

        return data.split("\n")
    except FileNotFoundError as e:
        print("Error opening and reading the file", e)
        return None


def clean_data(data: List[str]) -> List[int]:
    # remove blank lines
    data = list(filter(lambda line: line != " " and line != "", data))

    assert all(len(d) != 0 for d in data), "There's blank spaces not filtered out."

    # return only the lines indicating the selected clients
    data = list(
        filter(
            lambda line: "Best selected" in line
            or "Selected clients for this round: [" in line,
            data,
        )
    )

    # map lines to return the list of indices of best clients
    def group(line):
        pattern = r"([0-9]+)"

        indices = re.findall(pattern, line)
        return [int(i) for i in indices]

    data = list(map(group, data))

    assert all(all(isinstance(i, int) for i in selected) for selected in data)

    # flatten the indices
    data = reduce(lambda sublist, acc: sublist + acc, data, [])

    return data


def count_selections(selections: List[int]) -> dict:
    N_t = {}

    for key in selections:
        if key in N_t:
            N_t[key] += 1
        else:
            N_t[key] = 1

    return N_t


def fill_clients(N_t: dict, val: int, n_clients: int) -> dict:
    N_t = {k: N_t.get(k, val) for k in range(n_clients)}

    assert all(N_t[k] >= val for k in N_t)

    return N_t


def get_metrics(data: List[str]):
    # remove blank lines
    data = list(filter(lambda line: line != " " and line != "", data))

    assert all(len(d) != 0 for d in data), "There's blank spaces not filtered out."

    # return only the lines indicating the selected clients
    data = list(filter(lambda line: "Server:" in line, data))

    # map lines to return the list of metrics per round
    def group(line):
        pattern = r"([+-]?([0-9]*[.])?[0-9]+)"

        indices = re.findall(pattern, line)
        return [float(i[0]) for i in indices]

    data = list(map(group, data))

    assert all(all(isinstance(i, float) for i in m) for m in data)

    return data


def num_not_converged(train_out: List[str]):
    # filter not converged lines
    return len(list(filter(lambda line: "not converged" in line, train_out)))
