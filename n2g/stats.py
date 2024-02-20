import json
import pprint
import random
import typing
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Dict, List

import numpy as np
from pydantic import BaseModel, Field, PositiveInt


class ClassStats(BaseModel):
    precision: Annotated[float, Field(ge=0, le=1)]
    recall: Annotated[float, Field(ge=0, le=1)]
    f1_score: Annotated[float, Field(ge=0, le=1)]
    count: PositiveInt

    @staticmethod
    def _from_stats(stats: Dict[str, Any]) -> "ClassStats":
        return ClassStats(
            precision=stats["precision"],
            recall=stats["recall"],
            f1_score=stats["f1-score"],
            count=stats["support"],
        )

    def __post_init__(self) -> None:
        assert 0 <= self.precision <= 1, f"Precision should be between 0 and 1, but got {self.precision}"
        assert 0 <= self.recall <= 1, f"Recall should be between 0 and 1, but got {self.recall}"
        assert 0 <= self.f1_score <= 1, f"F1 score should be between 0 and 1, but got {self.f1_score}"
        assert 0 <= self.count, f"Count should be non-negative, but got {self.count}"

    def equal(self, other: "ClassStats") -> bool:
        """
        Check if two ClassStats objects are equal.
        Performs exact floating point comparison, so primarily useful for regression testing.
        """
        return (
            self.precision == other.precision
            and self.recall == other.recall
            and self.f1_score == other.f1_score
            and self.count == other.count
        )


class NeuronStats(BaseModel):
    accuracy: Annotated[float, Field(ge=0, le=1)]
    non_firing: ClassStats
    firing: ClassStats
    correlation: Annotated[float, Field(ge=-1, le=1)]

    @staticmethod
    def from_metrics_classification_report(report: Dict[str, Any], correlation: float) -> "NeuronStats":
        """
        Create NeuronStats from a classification report from the metrics package and a correlation value.

        Args:
            report: The classification report from the metrics package.
            correlation: The correlation value for the neuron.
        """
        return NeuronStats(
            accuracy=report["accuracy"],
            non_firing=ClassStats._from_stats(report["non_firing"]),
            firing=ClassStats._from_stats(report["firing"]),
            correlation=correlation,
        )

    def __post_init__(self) -> None:
        assert 0 <= self.accuracy <= 1, f"Accuracy should be between 0 and 1, but got {self.accuracy}"
        assert -1 <= self.correlation <= 1, f"Correlation should be between -1 and 1, but got {self.correlation}"

    def equal(self, other: "NeuronStats") -> bool:
        """
        Check if two NeuronStats objects are equal.
        Performs exact floating point comparison, so primarily useful for regression testing.
        """
        return (
            self.accuracy == other.accuracy
            and self.non_firing.equal(other.non_firing)
            and self.firing.equal(other.firing)
            and self.correlation == other.correlation
        )


def dump_neuron_stats(stats_path: Path, stats: Dict[int, Dict[int, NeuronStats]]) -> None:
    with open(stats_path, "w") as ofh:
        json.dump(
            {
                str(layer_index): {
                    str(neuron_index): neuron_stats.model_dump() for neuron_index, neuron_stats in stats.items()
                }
                for layer_index, stats in stats.items()
            },
            ofh,
        )


def load_neuron_stats(stats_path: Path) -> Dict[int, Dict[int, NeuronStats]] | None:
    if stats_path.exists():
        with open(stats_path) as ifh:
            loaded = json.load(ifh)
            return {
                int(layer_index): {
                    int(neuron_index): NeuronStats.model_validate(stats) for neuron_index, stats in neuron_stats.items()
                }
                for layer_index, neuron_stats in loaded.items()
            }
    else:
        return None


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
