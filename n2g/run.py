import random
import traceback
import typing
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split  # type: ignore
from torch import device
from transformer_lens.HookedTransformer import HookedTransformer  # type: ignore[import]
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer  # type: ignore

from . import scrape, train_and_eval
from .augmenter import AugmentationConfig, Augmenter, WordToCasings
from .fit import FitConfig, ImportanceConfig, PruneConfig
from .neuron_model import NeuronModel
from .neuron_store import NeuronStore
from .word_tokenizer import WordTokenizer


@dataclass
class TrainConfig:
    fit_config: FitConfig
    fire_threshold: float
    train_proportion: float
    random_seed: int

    def __init__(
        self,
        fit_config: Optional[FitConfig] = None,
        fire_threshold: float = 0.5,
        train_proportion: float = 0.5,
        random_seed: int = 0,
    ) -> None:
        if fit_config is None:
            self.fit_config = FitConfig(
                prune_config=PruneConfig(),
                importance_config=ImportanceConfig(),
                augmentation_config=AugmentationConfig(),
            )
        else:
            self.fit_config = fit_config
        self.fire_threshold = fire_threshold
        self.train_proportion = train_proportion
        self.random_seed = random_seed


def run_training(
    model: HookedTransformer,
    layer_indices: List[int],
    neuron_indices: List[int],
    layer_ending: str,
    augmenter: Augmenter,
    model_name: str,
    neuron_store: NeuronStore,
    all_stats: Dict[int, Dict[int, Dict[str, Any]]],
    config: TrainConfig,
) -> Tuple[Dict[int, Dict[int, NeuronModel]], NeuronStore, Dict[int, Dict[int, Dict[str, Any]]]]:
    random.seed(config.random_seed)

    neuron_models: Dict[int, Dict[int, NeuronModel]] = {}

    for layer_index in layer_indices:
        all_stats[layer_index] = {}
        neuron_models[layer_index] = {}
        for neuron_index in neuron_indices:
            print(f"{layer_index=} {neuron_index=}", flush=True)
            try:
                samples, base_max_activation = scrape.scrape_neuron(model_name, layer_index, neuron_index)

                split: Tuple[List[str], List[str]] = train_test_split(  # type: ignore
                    samples, train_size=config.train_proportion, random_state=config.random_seed
                )
                train_samples, test_samples = split

                neuron_model, stats = train_and_eval.train_and_eval(
                    model,
                    layer_index,
                    neuron_index,
                    augmenter,
                    train_samples,
                    test_samples,
                    base_max_activation,
                    layer_ending,
                    neuron_store,
                    fire_threshold=config.fire_threshold,
                    fit_config=config.fit_config,
                )

                neuron_models[layer_index][neuron_index] = neuron_model
                all_stats[layer_index][neuron_index] = stats

            except Exception:
                print(traceback.format_exc(), flush=True)
                print("Failed", flush=True)

    return neuron_models, neuron_store, all_stats


def run(
    model_name: str,
    layer_indices: List[int],
    neuron_indices: List[int],
    layer_ending: str,
    word_to_casings: WordToCasings,
    aug_model_name: str,
    neuron_store: NeuronStore,
    all_stats: Dict[int, Dict[int, Dict[str, Any]]],
    config: TrainConfig,
    device: device,
) -> Tuple[Dict[int, Dict[int, NeuronModel]], NeuronStore, Dict[int, Dict[int, Dict[str, Any]]]]:
    model: HookedTransformer = HookedTransformer.from_pretrained(model_name).to(device)  # type: ignore

    aug_model: PreTrainedModel = typing.cast(
        PreTrainedModel,
        AutoModelForMaskedLM.from_pretrained(aug_model_name).to(  # type: ignore
            device
        ),
    )
    aug_tokenizer: PreTrainedTokenizer = typing.cast(
        PreTrainedTokenizer,
        AutoTokenizer.from_pretrained(aug_model_name),  # type: ignore
    )

    stick_tokens = {"'"}
    word_tokenizer = WordTokenizer(set(), stick_tokens)
    augmenter = Augmenter(aug_model, aug_tokenizer, word_tokenizer, word_to_casings, device)

    return run_training(
        model,
        layer_indices,
        neuron_indices,
        layer_ending,
        augmenter,
        model_name,
        neuron_store,
        all_stats,
        config,
    )
