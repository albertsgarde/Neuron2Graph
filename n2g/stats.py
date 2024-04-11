import json
import math
from pathlib import Path
from typing import Annotated, Any

from jaxtyping import Bool
from numpy import ndarray
from pydantic import BaseModel, Field


class ClassStats(BaseModel):
    precision: Annotated[float, Field(allow_inf_nan=True)]
    recall: Annotated[float, Field(ge=0, le=1)]
    f1_score: Annotated[float, Field(allow_inf_nan=True)]
    count: Annotated[int, Field(ge=0)]

    @staticmethod
    def _from_stats(stats: dict[str, Any]) -> "ClassStats":
        return ClassStats(
            precision=stats["precision"],
            recall=stats["recall"],
            f1_score=stats["f1-score"],
            count=stats["support"],
        )

    @staticmethod
    def from_trues(
        trues: Bool[ndarray, " num_samples*sample_length"], pred_trues: Bool[ndarray, " num_samples*sample_length"]
    ) -> "ClassStats":
        """
        Create ClassStats from arrays indicating which samples were this class
        and which were predicted to be this class.

        Args:
            trues: A boolean array indicating which samples were this class.
            pred_trues: A boolean array indicating which samples were predicted to be this class.
        """
        assert trues.shape == pred_trues.shape, (
            "trues and pred_trues should have the same shape. "
            f"trues shape: {trues.shape}  pred_trues shape: {pred_trues.shape}"
        )
        num_true_positives = (trues & pred_trues).sum()
        num_false_positives = (~trues & pred_trues).sum()
        num_false_negatives = (trues & ~pred_trues).sum()
        if num_true_positives == 0:
            if num_false_positives == 0:
                precision = float("NaN")
            else:
                precision = 0
            if num_false_negatives == 0:
                recall = 1
            else:
                recall = 0
        else:
            precision = num_true_positives / (num_true_positives + num_false_positives)
            recall = num_true_positives / (num_true_positives + num_false_negatives)
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return ClassStats(precision=precision, recall=recall, f1_score=f1_score, count=trues.sum())

    def __post_init__(self) -> None:
        assert (
            math.isnan(self.precision) or 0 <= self.precision <= 1
        ), f"Precision should be NaN or between 0 and 1, but got {self.precision}"
        assert 0 <= self.recall <= 1, f"Recall should be between 0 and 1, but got {self.recall}"
        assert (
            math.isnan(self.f1_score) or 0 <= self.f1_score <= 1
        ), f"F1 score should be NaN or between 0 and 1, but got {self.f1_score}"
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

    def better(self, other: "ClassStats") -> bool:
        """
        Check if self is better than other.
        """
        return (
            self.precision >= other.precision
            and self.recall >= other.recall
            and self.f1_score >= other.f1_score
            and self.count == other.count
        )


class NeuronStats(BaseModel):
    accuracy: Annotated[float, Field(ge=0, le=1)]
    non_firing: ClassStats
    firing: ClassStats

    @staticmethod
    def from_metrics_classification_report(report: dict[str, Any]) -> "NeuronStats":
        """
        Create NeuronStats from a classification report from the metrics package.

        Args:
            report: The classification report from the metrics package.
        """
        return NeuronStats(
            accuracy=report["accuracy"],
            non_firing=ClassStats._from_stats(report["non_firing"]),
            firing=ClassStats._from_stats(report["firing"]),
        )

    @staticmethod
    def from_dict(stats: dict[str, Any]) -> "NeuronStats":
        """
        Create NeuronStats from a dictionary.

        Args:
            stats: The dictionary.
        """
        return NeuronStats.model_validate(stats)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @staticmethod
    def from_firings(
        firings: Bool[ndarray, " num_samples*sample_length"], pred_firings: Bool[ndarray, " num_samples*sample_length"]
    ) -> "NeuronStats":
        """
        Create NeuronStats from firings and predicted firings.

        Args:
            firings: The actual firings.
            pred_firings: The predicted firings.
        """
        assert firings.shape == pred_firings.shape, (
            "firings and pred_firings should have the same shape. "
            f"firings shape: {firings.shape}  pred_firings shape: {pred_firings.shape}"
        )
        accuracy = (firings == pred_firings).sum() / firings.size
        return NeuronStats(
            accuracy=accuracy,
            non_firing=ClassStats.from_trues(~firings, ~pred_firings),
            firing=ClassStats.from_trues(firings, pred_firings),
        )

    def equal(self, other: "NeuronStats") -> bool:
        """
        Check if two NeuronStats objects are equal.
        Performs exact floating point comparison, so primarily useful for regression testing.
        """
        return (
            self.accuracy == other.accuracy
            and self.non_firing.equal(other.non_firing)
            and self.firing.equal(other.firing)
        )

    def better(self, other: "NeuronStats") -> bool:
        """
        Check if self is better than other.
        """
        return (
            self.accuracy > other.accuracy
            and self.firing.better(other.firing)
            and self.non_firing.better(other.non_firing)
        )


def dump_neuron_stats(stats_path: Path, stats: dict[int, dict[int, NeuronStats]]) -> None:
    stats_path.mkdir(parents=True, exist_ok=True)
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


def load_neuron_stats(stats_path: Path) -> dict[int, dict[int, NeuronStats]] | None:
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
