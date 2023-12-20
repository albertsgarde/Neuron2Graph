import re
import sys
from typing import List, Tuple
import requests
import json
import os
import multiprocessing as mp
import numpy as np

from functools import partial

act_parser = re.compile("<h4>Max Act: <b>")


def get_max_activations(model_name: str, layer: int, neuron: int) -> float:
    """Get the max activating dataset examples for a given neuron in a model"""
    base_url = f"https://neuroscope.io/{model_name}/{layer}/{neuron}.html"

    response = requests.get(base_url)
    webpage = response.text

    parts = act_parser.split(webpage)
    if len(parts) < 2:
        raise Exception("No activations found")
    return float(parts[1].split("</b>")[0])


def get_max_acts(model_name: str, layer_and_neurons: Tuple[int, List[int]]):
    layer, neurons = layer_and_neurons
    activations: List[float] = []
    for i, neuron in enumerate(neurons):
        if i % 50 == 0:
            print(f"\nLayer {layer}: {i} of {len(neurons)} complete", flush=True)
        try:
            activation = get_max_activations(model_name, layer, neuron)
            activations.append(activation)
        except:
            print(f"Neuron {neuron} in layer {layer} failed", flush=True)
            # Use the previous activation as a hack to get around failures
            activations.append(activations[-1])
    return activations


def cmd_arguments() -> Tuple[str, int, int, bool]:
    """
    Gets model name, number of layers and neurons per layer from cmd arguments if available.
    """
    args = sys.argv[1:]
    num_arguments = len(args)
    if num_arguments < 3:
        raise Exception("Not enough arguments")
    model_name = args[0]
    num_layers = int(args[1])
    neurons_per_layer = int(args[2])
    overwrite = (args[3].lower() != "false") if num_arguments >= 4 else True

    return model_name, num_layers, neurons_per_layer, overwrite


if __name__ == "__main__":
    """
    Instructions:
    Look at https://neuroscope.io/ for the model you want to scrape
    Set the number of layers and neurons appropriately
    """
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)))
    )

    model_name, layers, neurons, overwrite = cmd_arguments()

    output_path = os.path.join(base_path, f"data/activation_matrix-{model_name}.json")

    if not overwrite and os.path.exists(output_path):
        print("File already exists. Exiting...")
        sys.exit(0)

    # Uncomment and overwrite if you would rather specify the configuration here.
    # model_name = "gpt2-small"
    # layers = 24
    # neurons = 4096

    info = [(layer, [neuron for neuron in range(neurons)]) for layer in range(layers)]

    with mp.Pool(layers) as p:
        activation_matrix = p.map(partial(get_max_acts, model_name), info)

    activation_matrix_np = np.array(activation_matrix)

    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(output_path, "w") as ofh:
        json.dump(activation_matrix, ofh, indent=2, ensure_ascii=False)
