from collections import defaultdict
import json
import pprint
import random
from typing import Dict, List, Set
import typing
import numpy as np


def get_summary_stats(
    path: str, verbose: bool = True
) -> List[Dict[str, Dict[str, float]]]:
    summary_stats = []
    summary_stds = []

    with open(path) as ifh:
        stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = json.load(ifh)

    missing = 0

    random.seed(0)

    inelegible_count = 0

    precision_case = 0

    for layer, layer_stats in stats.items():
        eligible_neurons = set(
            [
                neuron
                for neuron, neuron_stats in layer_stats.items()
                if "1" in neuron_stats
            ]
        )

        aggr_stats_dict: Dict[str, Dict[str, List[float]]] = {
            "Inactivating": defaultdict(list),
            "Activating": defaultdict(list),
        }
        for neuron, neuron_stats in layer_stats.items():
            if neuron not in eligible_neurons:
                inelegible_count += 1
                continue

            aggr_stats_dict["Inactivating"]["Precision"].append(
                neuron_stats["0"]["precision"]
            )
            aggr_stats_dict["Inactivating"]["Recall"].append(
                neuron_stats["0"]["recall"]
            )
            aggr_stats_dict["Inactivating"]["F1"].append(neuron_stats["0"]["f1-score"])

            # If we didn't predict anything as activating, treat this as 100% precision rather than 0%
            if neuron_stats["0"]["recall"] == 1 and neuron_stats["1"]["recall"] == 0:
                precision_case += 1
                neuron_stats["1"]["precision"] = 1.0

            aggr_stats_dict["Activating"]["Precision"].append(
                neuron_stats["1"]["precision"]
            )
            aggr_stats_dict["Activating"]["Recall"].append(neuron_stats["1"]["recall"])
            aggr_stats_dict["Activating"]["F1"].append(neuron_stats["1"]["f1-score"])

        if verbose:
            print(
                "Neurons Evaluated:", len(aggr_stats_dict["Inactivating"]["Precision"])
            )

        avg_stats_dict: Dict[str, Dict[str, float]] = {
            "Inactivating": {},
            "Activating": {},
        }
        std_stats_dict: Dict[str, Dict[str, float]] = {
            "Inactivating": {},
            "Activating": {},
        }
        for token_type, inner_stats_dict in aggr_stats_dict.items():
            for stat_type, stat_arr in inner_stats_dict.items():
                avg_stats_dict[token_type][stat_type] = typing.cast(
                    float, round(np.mean(stat_arr), 3)
                )
                std_stats_dict[token_type][stat_type] = typing.cast(
                    float, round(np.std(stat_arr), 3)
                )

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
