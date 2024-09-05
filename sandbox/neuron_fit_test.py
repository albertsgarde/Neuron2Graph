import importlib
import pickle

import n2g

importlib.reload(n2g)
from n2g import NeuronModel

with open("all_info.pkl", "rb") as f:
    all_info = pickle.load(f)

activation_threshold: float = 0.5
importance_threshold: float = 0.75

neuron_model = NeuronModel(
    activation_threshold,
    importance_threshold,
)
neuron_model.fit(all_info)
