import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import torch

import n2g
from n2g.augmenter import WordToCasings
from n2g.neuron_store import NeuronStore


def cmd_arguments() -> Tuple[str, List[int], int]:
    """
    Gets model name, layers, and neurons from the command line arguments if available.
    Layer ending should be either "mid" for SoLU models or "post" for GeLU models.
    The "mlp.hook_" prefix is added automatically.
    Layers are given either as `start:end` or as a comma separated list of layer indices.
    """
    args = sys.argv[1:]
    num_arguments = len(args)
    if num_arguments < 4:
        raise Exception("Not enough arguments. Expected model name, layers, and neurons.")
    model_name = args[0]
    layers_arg = args[1]
    if ":" in layers_arg:
        layer_range = layers_arg.split(":")
        layers = list(range(int(layer_range[0]), int(layer_range[1])))
    else:
        layers = [int(layer_index_str) for layer_index_str in layers_arg.split(",")]
    neurons_per_layer = int(args[2])

    return model_name, layers, neurons_per_layer


def layer_ending_from_model_name(model_name: str) -> str:
    if "solu" in model_name:
        return "mid"
    else:
        return "post"


def main() -> None:
    model_name, layer_indices, neurons_per_layer = cmd_arguments()
    layer_ending = f"mlp.hook_{layer_ending_from_model_name(model_name)}"

    print(
        f"Running N2G for model {model_name} layers {layer_indices}. "
        "Using layer ending {layer_ending} and {neurons_per_layer} neurons per layer."
    )

    # ================ Setup ================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    repo_root = (Path(__file__) / ".." / "..").resolve()

    data_dir = repo_root / "data"

    word_to_casings_path = data_dir / "word_to_casings.json"
    if not word_to_casings_path.exists():
        raise Exception("`word_to_casings.json` not found in `data/`.")

    with open(word_to_casings_path, "r", encoding="utf-8") as ifh:
        word_to_casings: WordToCasings = json.load(ifh)
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
    all_stats = n2g.get_neuron_stats(stats_path)

    # ================ Run ================
    # Run training for the specified layers and neurons

    neuron_models, neuron_store, neuron_stats = n2g.run(
        model_name,
        layer_indices,
        list(range(neurons_per_layer)),
        layer_ending,
        word_to_casings,
        aug_model_name,
        neuron_store,
        all_stats,
        n2g.TrainConfig(),
        device,
    )

    for layer_index, layer_neurons in neuron_models.items():
        for neuron_index, neuron_model in layer_neurons.items():
            assert neuron_model.layer == layer_index, "Neuron model layer index doesn't match expected layer index"
            assert neuron_model.neuron == neuron_index, "Neuron model neuron index doesn't match expected neuron index"
            net = neuron_model.graphviz()

            file_path = os.path.join(graph_dir, f"{layer_index}_{neuron_index}")
            with open(file_path, "w") as f:
                f.write(net.source)

    neuron_store.save(neuron_store_path)
    with open(stats_path, "w") as ofh:
        json.dump(neuron_stats, ofh, indent=2)
    with open(output_dir / "stats.pkl", "wb") as file:
        pickle.dump(neuron_stats, file)

    n2g.get_summary_stats(stats_path)


if __name__ == "__main__":
    """
    Instructions:
    Download word_to_casings.json from the repo and put in data/
    Set layer_ending to "mlp.hook_mid" for SoLU models and "mlp.hook_post" for GeLU models
    Download from the repo or scrape (with scrape.py) the activation matrix for the model and put in data/
    Set model_name to the name of the model you want to run for
    Set the parameters in the run section as desired
    Run this file!

    It will create neuron graphs for the specified layers and neurons in neuron_graphs/model_name/folder_name
    It'll also save the stats for each neuron in neuron_graphs/model_name/folder_name/stats.json
    And it will save the neuron store in neuron_graphs/model_name/neuron_store.json
    """
    main()
