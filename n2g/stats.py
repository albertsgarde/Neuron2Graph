import json
import pprint
import random
import typing
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_neuron_stats(stats_path: Path) -> Dict[int, Dict[int, Dict[str, Any]]]:
    if stats_path.exists():
        with open(stats_path) as ifh:
            return json.load(ifh)
    else:
        return {}


def get_summary_stats(path: Path, verbose: bool = True) -> List[Dict[str, Dict[str, float]]]:
    summary_stats: List[Dict[str, Dict[str, float]]] = []
    summary_stds: List[Dict[str, Dict[str, float]]] = []

    with open(path) as ifh:
        stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = json.load(ifh)

    random.seed(0)

    inelegible_count = 0

    precision_case = 0

    for _layer, layer_stats in stats.items():
        eligible_neurons = set([neuron for neuron, neuron_stats in layer_stats.items() if "firing" in neuron_stats])

        aggr_stats_dict: Dict[str, Dict[str, List[float]]] = {
            "Non-firing": defaultdict(list),
            "Firing": defaultdict(list),
        }
        for neuron, neuron_stats in layer_stats.items():
            if neuron not in eligible_neurons:
                inelegible_count += 1
                continue

            aggr_stats_dict["Non-firing"]["Precision"].append(neuron_stats["non_firing"]["precision"])
            aggr_stats_dict["Non-firing"]["Recall"].append(neuron_stats["non_firing"]["recall"])
            aggr_stats_dict["Non-firing"]["F1"].append(neuron_stats["non_firing"]["f1-score"])

            # If we didn't predict anything as activating, treat this as 100% precision rather than 0%
            if neuron_stats["non_firing"]["recall"] == 1 and neuron_stats["firing"]["recall"] == 0:
                precision_case += 1
                neuron_stats["firing"]["precision"] = 1.0

            aggr_stats_dict["Firing"]["Precision"].append(neuron_stats["firing"]["precision"])
            aggr_stats_dict["Firing"]["Recall"].append(neuron_stats["firing"]["recall"])
            aggr_stats_dict["Firing"]["F1"].append(neuron_stats["firing"]["f1-score"])

        if verbose:
            print("Neurons Evaluated:", len(aggr_stats_dict["Non-firing"]["Precision"]))

        avg_stats_dict: Dict[str, Dict[str, float]] = {
            "Non-firing": {},
            "Firing": {},
        }
        std_stats_dict: Dict[str, Dict[str, float]] = {
            "Non-firing": {},
            "Firing": {},
        }
        for token_type, inner_stats_dict in aggr_stats_dict.items():
            for stat_type, stat_arr in inner_stats_dict.items():
                avg_stats_dict[token_type][stat_type] = typing.cast(float, round(np.mean(stat_arr), 3))
                std_stats_dict[token_type][stat_type] = typing.cast(float, round(np.std(stat_arr), 3))

        summary_stats.append(avg_stats_dict)
        summary_stds.append(std_stats_dict)

    if verbose:
        for summary, std_summary in zip(summary_stats, summary_stds):
            print("\n", flush=True)
            pprint.pprint(summary)
            pprint.pprint(std_summary)

        print(f"{inelegible_count=}", flush=True)
        print(f"{precision_case=}", flush=True)

    return summary_stats
