import re
import json
from typing import List, Tuple
import os
import random
import time
import sys
import traceback

import numpy as np

import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

import n2g
from n2g import FastAugmenter, WordTokenizer, NeuronStore, train_and_eval


def run_training(layers, neurons, sample_num=None, params=None, start_neuron=None):
    if params is None or not params:
        params = {
            "importance_threshold": 0.75,
            "n": 5,
            "max_train_size": None,
            "train_proportion": 0.5,
            "max_eval_size": 0.5,
            "activation_threshold": 0.5,
            "token_activation_threshold": 1,
            "fire_threshold": 0.5,
        }
    print(f"{params=}\n", flush=True)
    random.seed(0)

    all_neuron_indices = [i for i in range(neurons)]

    if not os.path.exists(f"{base_path}/neuron_graphs/{model_name}"):
        os.mkdir(f"{base_path}/neuron_graphs/{model_name}")

    neuron_store = NeuronStore(
        f"{base_path}/neuron_graphs/{model_name}/neuron_store.json"
    )

    folder_path = os.path.join(base_path, f"neuron_graphs/{model_name}")

    if not os.path.exists(folder_path):
        print("Making", folder_path, flush=True)
        os.mkdir(folder_path)

    if os.path.exists(f"{folder_path}/stats.json"):
        with open(f"{folder_path}/stats.json") as ifh:
            all_stats = json.load(ifh)

    else:
        all_stats = {}

    printerval = 10

    for layer in layers:
        t1 = time.perf_counter()

        if sample_num is None:
            chosen_neuron_indices = all_neuron_indices
        else:
            chosen_neuron_indices = random.sample(all_neuron_indices, sample_num)
            chosen_neuron_indices = sorted(chosen_neuron_indices)

        all_stats[layer] = {}
        for i, neuron in enumerate(chosen_neuron_indices):
            if start_neuron is not None and neuron < start_neuron:
                continue

            print(f"{layer=} {neuron=}", flush=True)
            try:
                stats = train_and_eval(
                    model,
                    layer,
                    neuron,
                    fast_aug,
                    base_path,
                    model_name,
                    activation_matrix,
                    layer_ending,
                    neuron_store,
                    **params,
                )

                all_stats[layer][neuron] = stats

                if i % printerval == 0:
                    t2 = time.perf_counter()
                    elapsed = t2 - t1
                    rate = printerval / elapsed
                    remaining = (len(chosen_neuron_indices) - i) / rate / 60
                    print(
                        f"{i} complete, batch took {elapsed / 60:.2f} mins, {rate=:.2f} neurons/s, {remaining=:.1f} mins"
                    )
                    t1 = t2

                    neuron_store.save()
                    with open(f"{folder_path}/stats.json", "w") as ofh:
                        json.dump(all_stats, ofh, indent=2)

            except Exception as e:
                print(traceback.format_exc(), flush=True)
                print("Failed", flush=True)

    neuron_store.save()
    with open(f"{folder_path}/stats.json", "w") as ofh:
        json.dump(all_stats, ofh, indent=2)


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

    model_name, layer_ending, layers, neurons_per_layer = cmd_arguments()
    # Uncomment and overwrite if you would rather specify the model name and layer ending here.
    # model_name = "gpt2-small"
    # layer_ending = "mlp.hook_post"

    print(
        f"Running N2G for model {model_name} layers {layers}. Using layer ending {layer_ending} and {neurons_per_layer} neurons per layer."
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

    # main()

    if not os.path.exists(f"{base_path}/neuron_graphs"):
        os.mkdir(f"{base_path}/neuron_graphs")

    # ================ Run ================
    # Run training for the specified layers and neurons

    # Override params as desired - sensible defaults are set in run_training
    params = {}

    run_training(
        # List of layers to run for
        layers=layers,
        # Number of neurons in each layer
        neurons=neurons_per_layer,
        # Neuron to start at (useful for resuming - None to start at 0)
        start_neuron=None,
        # Number of neurons to sample from each layer (None for all neurons)
        sample_num=None,
        params=params,
    )

    n2g.get_summary_stats(f"{base_path}/neuron_graphs/{model_name}/stats.json")
