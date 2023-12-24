import json
from typing import List, Tuple
import os
import sys

import numpy as np

import torch
from transformers import AutoModelForMaskedLM  # type: ignore
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

import n2g
from n2g import FastAugmenter, WordTokenizer


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

    model_name, layer_ending, layer_indices, neurons_per_layer = cmd_arguments()

    print(
        f"Running N2G for model {model_name} layers {layer_indices}. Using layer ending {layer_ending} and {neurons_per_layer} neurons per layer."
    )

    # ================ Setup ================
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    model = HookedTransformer.from_pretrained(model_name).to(device)

    base_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)))
    )

    # Save the activation matrix for the model to data/
    activation_matrix_path = os.path.join(
        base_path, f"data/activation_matrix-{model_name}.json"
    )
    if not os.path.exists(activation_matrix_path):
        raise Exception(
            f"Activation matrix not found for model {model_name}. Either download it from the repo or scrape it with `scrape.py`."
        )
    with open(activation_matrix_path) as ifh:
        activation_matrix = json.load(ifh)
        activation_matrix = np.array(activation_matrix)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    aug_model_checkpoint = "distilbert-base-uncased"
    aug_model = AutoModelForMaskedLM.from_pretrained(aug_model_checkpoint).to(device)
    aug_tokenizer = AutoTokenizer.from_pretrained(aug_model_checkpoint)

    with open(f"{base_path}/data/word_to_casings.json", encoding="utf-8") as ifh:
        word_to_casings = json.load(ifh)

    stick_tokens = {"'"}
    word_tokenizer = WordTokenizer(set(), stick_tokens)
    fast_aug = FastAugmenter(aug_model, aug_tokenizer, word_tokenizer, word_to_casings)

    output_dir = os.path.join(base_path, "output", model_name)

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
        fast_aug,
        # Activation matrix for the model
        activation_matrix,
        # Model name
        model_name,
        # Base path
        output_dir,
    )

    n2g.get_summary_stats(os.path.join(output_dir, "stats.json"))
