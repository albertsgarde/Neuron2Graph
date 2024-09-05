import json
import os
from pathlib import Path
from typing import Dict

import torch

import n2g
from n2g import stats
from n2g.neuron_store import NeuronStore
from n2g.stats import NeuronStats


def main() -> None:
    model_name = "solu-2l"
    layer_ending = "mlp.hook_mid"
    layer_indices = [0, 1]
    neuron_indices = [0, 1, 2, 3, 4]

    # model_name = "gelu-1l"
    # layer_ending = "mlp.hook_post"
    # layer_indices = [0]
    # neuron_indices = [0, 338]

    print(
        f"Running N2G for model {model_name} layers {layer_indices}. "
        f"Using layer ending {layer_ending} and neurons {neuron_indices}."
    )

    # ================ Setup ================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    repo_root = (Path(__file__) / ".." / "..").resolve()

    data_dir = repo_root / "data"

    word_to_casings_path = data_dir / "word_to_casings.json"
    if not word_to_casings_path.exists():
        raise Exception("`word_to_casings.json` not found in `data/`.")

    with open(word_to_casings_path, encoding="utf-8") as ifh:
        word_to_casings = json.load(ifh)
    aug_model_name = "distilbert-base-uncased"

    output_dir = repo_root / "output" / model_name
    os.makedirs(output_dir, exist_ok=True)

    graph_dir = output_dir / "graphs"
    neuron_store_path = output_dir / "neuron_store.json"
    stats_path = output_dir / "stats.json"
    # This ensures that both the base output path and the graph directory exist
    if not graph_dir.exists():
        os.makedirs(graph_dir)

    neuron_store = NeuronStore.load(neuron_store_path) if neuron_store_path.exists() else NeuronStore()
    all_stats = stats.load_neuron_stats(stats_path)
    if all_stats is None:
        all_stats = {}

    # ================ Run ================
    # Run training for the specified layers and neurons

    neuron_models, _neuron_store, all_stats = n2g.run(
        model_name,
        layer_indices,
        neuron_indices,
        layer_ending,
        word_to_casings,
        aug_model_name,
        neuron_store,
        all_stats,
        n2g.TrainConfig(),
        device,
    )

    baseline_stats_path = repo_root / "output" / "solu-2l-baseline" / "stats.json"
    baseline_stats: Dict[int, Dict[int, NeuronStats]] = stats.load_neuron_stats(baseline_stats_path)  # type: ignore
    assert baseline_stats != {}, "No stats found. Make sure to run a baseline first."

    worse_neurons = []
    better_neurons = []
    total_f1_diff = 0.0
    total_recall_diff = 0.0
    total_precision_diff = 0.0

    # Compare stats
    assert (
        all_stats.keys() == baseline_stats.keys()
    ), f"Layer indices don't match. stats {all_stats.keys()} compare_stats {baseline_stats.keys()}"
    for (layer_index, layer_neurons), (comp_layer_index, comp_layer_neurons) in zip(
        all_stats.items(), baseline_stats.items()
    ):
        assert (
            layer_index == comp_layer_index
        ), f"Layer indices don't match. Layer {layer_index}  comp_layer_index {comp_layer_index}"
        assert layer_neurons.keys() == comp_layer_neurons.keys(), (
            f"Neuron indices don't match for layer {layer_index}"
            f"layer_neurons {layer_neurons.keys()} comp_layer_neurons {comp_layer_neurons.keys()}"
        )
        for (neuron_index, neuron_stats), (comp_neuron_index, comp_neuron_stats) in zip(
            layer_neurons.items(), comp_layer_neurons.items()
        ):
            assert neuron_index == comp_neuron_index
            neuron_stats.equal(comp_neuron_stats)
            if neuron_stats.better(comp_neuron_stats):
                better_neurons.append(
                    {
                        "layer": layer_index,
                        "neuron": neuron_index,
                        "stats": neuron_stats.to_dict(),
                        "comp_stats": comp_neuron_stats.to_dict(),
                    }
                )
            else:
                worse_neurons.append(
                    {
                        "layer": layer_index,
                        "neuron": neuron_index,
                        "stats": neuron_stats.to_dict(),
                        "comp_stats": comp_neuron_stats.to_dict(),
                    }
                )
            total_f1_diff += neuron_stats.firing.f1_score - comp_neuron_stats.firing.f1_score
            total_recall_diff += neuron_stats.firing.recall - comp_neuron_stats.firing.recall
            total_precision_diff += neuron_stats.firing.precision - comp_neuron_stats.firing.precision

    with open("comp.json", "w", encoding="utf-8") as f:
        num_neurons = len(better_neurons) + len(worse_neurons)
        result = {
            "avg_f1_diff": total_f1_diff / num_neurons,
            "total_recall_diff": total_recall_diff / num_neurons,
            "total_precision_diff": total_precision_diff / num_neurons,
            "num_better_neurons": len(better_neurons),
            "num_worse_neurons": len(worse_neurons),
            "better_neurons": better_neurons,
            "worse_neurons": worse_neurons,
        }
        json.dump(result, f)

    # Load baseline graphs by loading all existing graphs in the baseline graph directory
    baseline_graph_dir = repo_root / "output" / "solu-2l-baseline" / "graphs"
    baseline_graphs: dict[int, dict[int, str]] = {}
    for graph_file in baseline_graph_dir.iterdir():
        with open(graph_file, "r") as f:
            [layer_index, neuron_index] = [int(part) for part in graph_file.stem.split("_")]
            if layer_index not in baseline_graphs:
                baseline_graphs[layer_index] = {}
            baseline_graphs[layer_index][neuron_index] = f.read()

    assert len(baseline_graphs) == len(neuron_models), (
        "Number of layers don't match" f"layers {baseline_graphs.keys()} baseline_layers {neuron_models.keys()}"
    )
    for layer_index, layer_models in neuron_models.items():
        assert layer_index in baseline_graphs, f"Layer {layer_index} not found in baseline graphs"
        assert len(layer_models) == len(baseline_graphs[layer_index]), (
            f"Number of neurons don't match for layer {layer_index}"
            f"layer_models {layer_models} baseline_graphs {baseline_graphs[layer_index]}"
        )
        for neuron_index, _neuron_model in layer_models.items():
            assert (
                neuron_index in baseline_graphs[layer_index]
            ), f"Neuron {neuron_index} in layer {layer_index} not found in baseline graphs"
            # model_graph = neuron_model.graphviz()
            # baseline_graph = baseline_graphs[layer_index][neuron_index]
    #
    # mismatch = next(
    #    (
    #        (i, c1, c2)
    #        for i, (c1, c2) in enumerate(zip("".join(model_graph.split()), "".join(baseline_graph.split())))
    #        if c1 != c2
    #    ),
    #    None,
    # )
    # assert mismatch is None, (
    #    f"Graphs don't match for l{layer_index}n{neuron_index}\n"
    #    f"First mismatch at index {mismatch[0]} (ignoring whitespace): {mismatch[1]} != {mismatch[2]}\n"
    #    f"neuron_model graph:\n{model_graph}\n baseline graph:\n{baseline_graph}"
    # )

    print("Test passed")


if __name__ == "__main__":
    main()
