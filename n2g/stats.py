import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, PositiveInt


class ClassStats(BaseModel):
    precision: Annotated[float, Field(ge=0, le=1)]
    recall: Annotated[float, Field(ge=0, le=1)]
    f1_score: Annotated[float, Field(ge=0, le=1)]
    count: PositiveInt

    @staticmethod
    def _from_stats(stats: dict[str, Any]) -> "ClassStats":
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
    def from_metrics_classification_report(report: dict[str, Any], correlation: float) -> "NeuronStats":
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


def dump_neuron_stats(stats_path: Path, stats: dict[int, dict[int, NeuronStats]]) -> None:
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
