import json
import os
import pickle
from pathlib import Path
from typing import Dict
import time

import torch

import n2g
from n2g import stats
from n2g.feature_model import FeatureModel
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
    all_stats: dict[int, dict[int, NeuronStats]] = {}

    # ================ Run ================
    # Run training for the specified layers and neurons

    n2g_start_time = time.time()
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
    n2g_end_time = time.time()
    print(f"N2G time: {n2g_end_time - n2g_start_time} seconds")

    stats.dump_neuron_stats(stats_path, all_stats)

    import transformer_lens

    model = transformer_lens.HookedTransformer.from_pretrained(model_name).to(device)  # type: ignore
    tokenizer = n2g.Tokenizer(model)

    graph_dir.mkdir(exist_ok=True, parents=True)
    for layer_index, layer_models in neuron_models.items():
        for neuron_index, neuron_model in layer_models.items():
            with (graph_dir / f"{layer_index}_{neuron_index}.pkl").open("wb") as f:
                model_bytes = FeatureModel.from_model(tokenizer, neuron_model).to_bytes()
                print(f"Model size: {len(model_bytes)}")
                f.write(model_bytes)
            with (graph_dir / f"{layer_index}_{neuron_index}.dot").open("w") as f:
                f.write(neuron_model.graphviz().source)
    all_models_bytes = FeatureModel.list_to_bin(
        [
            FeatureModel.from_model(tokenizer, neuron_model)
            for layer_models in neuron_models.values()
            for neuron_model in layer_models.values()
        ]
    )
    with (graph_dir / "models.pkl").open("wb") as f:
        print(f"All models size: {len(all_models_bytes)}")
        f.write(all_models_bytes)

    baseline_stats_path = repo_root / "output" / f"{model_name}-baseline" / "stats.json"
    baseline_stats: Dict[int, Dict[int, NeuronStats]] = stats.load_neuron_stats(baseline_stats_path)  # type: ignore
    assert baseline_stats != {}, "No stats found. Make sure to run a baseline first."

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
            assert neuron_stats.equal(comp_neuron_stats), (
                f"Neuron stats don't match for l{layer_index}n{neuron_index}\n"
                f"neuron_stats {neuron_stats} comp_neuron_stats {comp_neuron_stats}"
            )

    # Load baseline graphs by loading all existing graphs in the baseline graph directory
    baseline_graph_dir = repo_root / "output" / "solu-2l-baseline" / "graphs"
    baseline_graphs: dict[int, dict[int, str]] = {}

    for graph_file in baseline_graph_dir.iterdir():
        if graph_file.suffix == ".dot":
            with open(graph_file, "r") as f:
                [layer_index, neuron_index] = [int(part) for part in graph_file.stem.split("_")]
                if layer_index not in baseline_graphs:
                    baseline_graphs[layer_index] = {}
                baseline_graphs[layer_index][neuron_index] = f.read()

    assert len(baseline_graphs) == len(
        neuron_models
    ), f"Number of layers ({baseline_graphs.keys()}) doesn¨t match baseline number of layers ({neuron_models.keys()})"
    for layer_index, layer_models in neuron_models.items():
        assert layer_index in baseline_graphs, f"Layer {layer_index} not found in baseline graphs"
        assert len(layer_models) == len(baseline_graphs[layer_index]), (
            f"Number of neurons don't match for layer {layer_index}"
            f"layer_models {layer_models} baseline_graphs {baseline_graphs[layer_index]}"
        )
        for neuron_index, neuron_model in layer_models.items():
            assert (
                neuron_index in baseline_graphs[layer_index]
            ), f"Neuron {neuron_index} in layer {layer_index} not found in baseline graphs"
            baseline_neuron_graph = baseline_graphs[layer_index][neuron_index]
            assert (
                neuron_model.graphviz().source == baseline_neuron_graph
            ), f"Graphs don't match for l{layer_index}n{neuron_index}\n"

    total_model_file_size = 0
    total_baseline_model_file_size = 0
    for layer_index, layer_models in neuron_models.items():
        for neuron_index, _neuron_model in layer_models.items():
            model_file = graph_dir / f"{layer_index}_{neuron_index}.pkl"
            assert model_file.exists(), f"Model file {model_file} not found"
            baseline_model_file = baseline_graph_dir / f"{layer_index}_{neuron_index}.pkl"
            assert baseline_model_file.exists(), f"Baseline model file {baseline_model_file} not found"
            model_file_size = model_file.stat().st_size
            baseline_model_file_size = baseline_model_file.stat().st_size
            total_model_file_size += model_file_size
            total_baseline_model_file_size += baseline_model_file_size

    print(f"Total model file size: {total_model_file_size/1000} KB")
    print(f"Total baseline model file size: {total_baseline_model_file_size/1000} KB")
    print("Test passed")


if __name__ == "__main__":
    main()
