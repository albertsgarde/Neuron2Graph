import json
import re
from typing import List, Tuple

import requests

parser = re.compile('\\{"tokens": ')


act_parser = re.compile("<h4>Max Act: <b>")


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
