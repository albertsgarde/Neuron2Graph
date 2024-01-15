import json
import os
import sys
import typing
from typing import List, Tuple

import numpy as np
import torch
from transformer_lens.HookedTransformer import HookedTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer  # type: ignore

import n2g
from n2g import Augmenter, WordTokenizer


def cmd_arguments() -> Tuple[str, str, List[int], int]:
    """
    Gets model name, layer ending, layers, and neurons from the command line arguments if available.
    Layer ending should be either "mid" for SoLU models or "post" for GeLU models.
    The "mlp.hook_" prefix is added automatically.
    Layers are given either as `start:end` or as a comma separated list of layer indices.
    """
    args = sys.argv[1:]
    num_arguments = len(args)
    if num_arguments < 4:
        raise Exception("Not enough arguments")
    model_name = args[0]
    layer_ending = f"mlp.hook_{args[1]}"
    layers_arg = args[2]
    if ":" in layers_arg:
        layer_range = layers_arg.split(":")
        layers = list(range(int(layer_range[0]), int(layer_range[1])))
    else:
        layers = [int(layer_index_str) for layer_index_str in layers_arg.split(",")]
    neurons_per_layer = int(args[3])

    return model_name, layer_ending, layers, neurons_per_layer


def main() -> None:
    model_name, layer_ending, layer_indices, neurons_per_layer = cmd_arguments()

    print(
        f"Running N2G for model {model_name} layers {layer_indices}. Using layer ending {layer_ending} and {neurons_per_layer} neurons per layer."
    )

    # ================ Setup ================
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    model: HookedTransformer = HookedTransformer.from_pretrained(model_name).to(device)  # type: ignore

    repo_root: str = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Save the activation matrix for the model to data/
    activation_matrix_path = os.path.join(repo_root, f"data/activation_matrix-{model_name}.json")
    if not os.path.exists(activation_matrix_path):
        raise Exception(
            f"Activation matrix not found for model {model_name}. "
            "Either download it from the repo or scrape it with `scrape.py`."
        )
    with open(activation_matrix_path) as ifh:
        activation_matrix = json.load(ifh)
        activation_matrix = np.array(activation_matrix)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    aug_model_checkpoint = "distilbert-base-uncased"
    aug_model: PreTrainedModel = typing.cast(
        PreTrainedModel,
        AutoModelForMaskedLM.from_pretrained(aug_model_checkpoint).to(  # type: ignore
            device
        ),
    )
    aug_tokenizer: PreTrainedTokenizer = typing.cast(
        PreTrainedTokenizer,
        AutoTokenizer.from_pretrained(aug_model_checkpoint),  # type: ignore
    )

    data_dir = os.path.join(repo_root, "data")
    if not os.path.exists(os.path.join(data_dir, "word_to_casings.json")):
        raise Exception("`word_to_casings.json` not found in `data/`.")

    with open(os.path.join(data_dir, "word_to_casings.json"), encoding="utf-8") as ifh:
        word_to_casings = json.load(ifh)

    stick_tokens = {"'"}
    word_tokenizer = WordTokenizer(set(), stick_tokens)
    augmenter = Augmenter(aug_model, aug_tokenizer, word_tokenizer, word_to_casings)

    output_dir = os.path.join(repo_root, "output", model_name)
    os.makedirs(output_dir, exist_ok=True)

    # ================ Run ================
    # Run training for the specified layers and neurons

    n2g.run_training(
        model,
        # List of layers to run for
        layer_indices,
        # Number of neurons in each layer
        list(range(neurons_per_layer)),
        # Layer ending for the model
        layer_ending,
        # Augmenter
        augmenter,
        # Activation matrix for the model
        activation_matrix,
        # Model name
        model_name,
        # Base path
        output_dir,
    )

    n2g.get_summary_stats(os.path.join(output_dir, "stats.json"))


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
