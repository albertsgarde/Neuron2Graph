import itertools
import json
import sys
from pathlib import Path
from typing import Callable, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer

import n2g
from n2g import NeuronStats, scrape


def main() -> None:
    overwrite: bool = len(sys.argv) > 1 and sys.argv[1] == "-o"

    model_name = "solu-2l"
    num_features = 5

    # ================ Setup ================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    repo_root = (Path(__file__) / ".." / "..").resolve()

    word_to_casings_path = repo_root / "data" / "word_to_casings.json"

    if not word_to_casings_path.exists():
        raise Exception("`word_to_casings.json` not found in `data/`.")
    with open(word_to_casings_path, encoding="utf-8") as ifh:
        word_to_casings = json.load(ifh)

    model = HookedTransformer.from_pretrained(model_name, device=device)

    tokenizer = n2g.Tokenizer(model)

    # ================ Run ================
    # Run training for the specified layers and neurons

    def activation(
        layer_index: int,
    ) -> Callable[
        [int], Callable[[Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]]
    ]:
        layer_id: str = f"blocks.{layer_index}.mlp.hook_mid"
        return lambda neuron_index: n2g.feature_activation(model, layer_id, neuron_index)

    def samples(layer_index: int) -> Callable[[int], Tuple[list[str], float]]:
        return lambda neuron_index: scrape.scrape_neuron(model_name, layer_index, neuron_index)

    new_neuron_stats = list(
        itertools.chain.from_iterable(
            [
                n2g.run_layer(
                    num_features,
                    activation(layer_index),
                    samples(layer_index),
                    tokenizer,
                    word_to_casings,
                    device,
                    n2g.TrainConfig(),
                )[1]
                for layer_index in range(2)
            ]
        )
    )

    baseline_stats_path: Path = repo_root / "output" / "solu-2l-stats-baseline.json"

    if overwrite:
        with baseline_stats_path.open("w", encoding="utf-8") as file:
            json_object = [neuron_stats.model_dump() for neuron_stats in new_neuron_stats]
            json.dump(json_object, file)
        print("New baseline created.")
    else:
        with baseline_stats_path.open("r", encoding="utf-8") as file:
            json_object = json.load(file)
            baseline_stats = [NeuronStats.model_validate(stats) for stats in json_object]
        assert len(baseline_stats) == len(new_neuron_stats)
        baseline_avg_f1_score = sum(stats.firing.f1_score for stats in baseline_stats) / len(baseline_stats)
        new_avg_f1_score = sum(stats.firing.f1_score for stats in new_neuron_stats) / len(new_neuron_stats)
        assert new_avg_f1_score >= baseline_avg_f1_score

        regressions = set()

        def avg_delta(
            new: list[NeuronStats], baseline: list[NeuronStats], field: Callable[[NeuronStats], float]
        ) -> float:
            return sum(
                field(new) - field(baseline) for new, baseline in zip(new_neuron_stats, baseline_stats, strict=True)
            ) / len(new_neuron_stats)

        print(
            "Average firing precision delta: "
            f"{avg_delta(new_neuron_stats, baseline_stats, lambda x: x.firing.precision)}"
        )
        print(
            "Average firing recall delta: " f"{avg_delta(new_neuron_stats, baseline_stats, lambda x: x.firing.recall)}"
        )
        print(
            "Average firing f1 score delta: "
            f"{avg_delta(new_neuron_stats, baseline_stats, lambda x: x.firing.f1_score)}"
        )
        print(
            "Average non-firing precision delta: "
            f"{avg_delta(new_neuron_stats, baseline_stats, lambda x: x.non_firing.precision)}"
        )
        print(
            "Average non-firing recall delta: "
            f"{avg_delta(new_neuron_stats, baseline_stats, lambda x: x.non_firing.recall)}"
        )
        print(
            "Average non-firing f1 score delta: "
            f"{avg_delta(new_neuron_stats, baseline_stats, lambda x: x.non_firing.f1_score)}"
        )

        for i, (baseline, new) in enumerate(zip(baseline_stats, new_neuron_stats, strict=True)):
            messages = []
            if new.accuracy < baseline.accuracy - 0.01:
                messages += f"    Accuracy decreased from {baseline.accuracy} to {new.accuracy}."
                regressions.add(i)
            if new.non_firing.precision < baseline.non_firing.precision - 0.01:
                messages += (
                    f"    Non-firing precision decreased from {baseline.non_firing.precision} "
                    f"to {new.non_firing.precision}."
                )
                regressions.add(i)
            if new.non_firing.recall < baseline.non_firing.recall - 0.01:
                messages += (
                    f"    Non-firing recall decreased from {baseline.non_firing.recall} to {new.non_firing.recall}."
                )
                regressions.add(i)
            if new.firing.precision < baseline.firing.precision - 0.01:
                messages += (
                    f"    Firing precision decreased from {baseline.firing.precision} to {new.firing.precision}."
                )
                regressions.add(i)
            if new.firing.recall < baseline.firing.recall - 0.01:
                messages += f"    Firing recall decreased from {baseline.firing.recall} to {new.firing.recall}."
                regressions.add(i)
            if messages != []:
                print(f"Regression found for neuron {i}:\n" + "\n".join(messages))

        if regressions == set():
            print("Test passed.")
        else:
            print(f"Test failed. Regressions found for features {regressions}.")
    exit()

    print("Test passed")


if __name__ == "__main__":
    main()
