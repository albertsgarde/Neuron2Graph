import json
import re
from typing import List, Tuple

import requests

parser = re.compile('\\{"tokens": ')


def scrape_neuroscope_samples(model_name: str, layer: int, neuron: int) -> List[str]:
    """Get the max activating dataset examples for a given neuron in a model"""
    base_url = f"https://neuroscope.io/{model_name}/{layer}/{neuron}.html"

    response = requests.get(base_url)
    webpage = response.text

    parts = parser.split(webpage)
    snippets: List[str] = []
    for i, part in enumerate(parts):
        if i == 0 or i % 2 != 0:
            continue

        token_str = part.split(', "values": ')[0]

        tokens = json.loads(token_str)

        snippet = "".join(tokens)

        snippets.append(snippet)

    if len(snippets) != 20:
        raise Exception
    return snippets


act_parser = re.compile("<h4>Max Act: <b>")


def get_max_activation(model_name: str, layer: int, neuron: int) -> float:
    """Get the max activating dataset examples for a given neuron in a model"""
    base_url = f"https://neuroscope.io/{model_name}/{layer}/{neuron}.html"

    response = requests.get(base_url)
    webpage = response.text

    parts = act_parser.split(webpage)
    if len(parts) < 2:
        raise Exception("No activations found")
    return float(parts[1].split("</b>")[0])


def scrape_neuron(model_name: str, layer: int, neuron: int) -> Tuple[List[str], float]:
    base_url = f"https://neuroscope.io/{model_name}/{layer}/{neuron}.html"

    response = requests.get(base_url)
    webpage = response.text

    parts = act_parser.split(webpage)
    if len(parts) < 2:
        raise Exception("No activations found")
    max_activation = float(parts[1].split("</b>")[0])

    snippets: List[str] = []
    for i, part in enumerate(parser.split(webpage)):
        if i == 0 or i % 2 != 0:
            continue

        token_str = part.split(', "values": ')[0]

        tokens = json.loads(token_str)

        snippet = "".join(tokens)

        snippets.append(snippet)

    if len(snippets) != 20:
        raise Exception
    return snippets, max_activation


def get_max_acts(model_name: str, layer_and_neurons: Tuple[int, List[int]]):
    layer, neurons = layer_and_neurons
    activations: List[float] = []
    for i, neuron in enumerate(neurons):
        if i % 50 == 0:
            print(f"\nLayer {layer}: {i} of {len(neurons)} complete", flush=True)
        try:
            activation = get_max_activation(model_name, layer, neuron)
            activations.append(activation)
        except Exception:
            print(f"Neuron {neuron} in layer {layer} failed", flush=True)
            # Use the previous activation as a hack to get around failures
            activations.append(activations[-1])
    return activations
