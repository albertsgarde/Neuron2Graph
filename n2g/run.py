import random
import sys
import traceback
import typing
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from jaxtyping import Float, Int
from sklearn.model_selection import train_test_split  # type: ignore
from torch import Tensor, device
from transformer_lens import HookedTransformer  # type: ignore[import]
from transformer_lens.hook_points import HookPoint  # type: ignore[import]
from transformers import (  # type: ignore[import]
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from n2g.stats import NeuronStats  # type: ignore

from . import scrape, train_and_eval
from .augmenter import AugmentationConfig, Augmenter, WordToCasings
from .fit import FitConfig, ImportanceConfig, PruneConfig
from .neuron_model import NeuronModel
from .neuron_store import NeuronStore
from .tokenizer import Tokenizer
from .word_tokenizer import WordTokenizer


@dataclass
class TrainConfig:
    fit_config: FitConfig
    fire_threshold: float
    train_proportion: float
    random_seed: int
    stop_on_error: bool

    def __init__(
        self,
        fit_config: Optional[FitConfig] = None,
        fire_threshold: float = 0.5,
        train_proportion: float = 0.5,
        random_seed: int = 0,
        stop_on_error: bool = True,
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
        self.stop_on_error = stop_on_error


def layer_index_to_name(layer_index: int, layer_ending: str) -> str:
    return f"blocks.{layer_index}.{layer_ending}"


def feature_activation(
    model: HookedTransformer, layer_id: str, neuron_index: int
) -> Callable[[Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]]:
    def result(samples: Int[Tensor, "num_samples sample_length"]) -> Float[Tensor, "num_samples sample_length"]:
        activations: Float[Tensor, "num_samples sample_length"] = torch.full(samples.shape, float("nan"))

        def hook(activation: Float[Tensor, "num_samples sample_length neurons_per_layer"], hook: HookPoint) -> None:
            activations[:] = activation[:, :, neuron_index]

        with torch.no_grad():
            model.run_with_hooks(samples, fwd_hooks=[(layer_id, hook)])
            assert not torch.isnan(activations).any(), "Activations should not contain NaNs"

        return activations

    return result


def default_augmenter(word_to_casings: WordToCasings, device: device) -> Augmenter:
    aug_model_name = "distilbert-base-uncased"
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
    return Augmenter(aug_model, aug_tokenizer, word_tokenizer, word_to_casings, device)


def run_layer(
    feature_indices: Sequence[int],
    feature_activation: Callable[
        [int], Callable[[Int[Tensor, "num_samples sample_length"]], Float[Tensor, "num_samples sample_length"]]
    ],
    feature_samples: Callable[[int], Tuple[list[str], float]],
    tokenizer: Tokenizer,
    word_to_casings: WordToCasings,
    device: device,
    train_config: TrainConfig,
) -> Tuple[list[NeuronModel | None], list[NeuronStats | None]]:
    feature_models: list[NeuronModel | None] = []
    feature_stats: list[NeuronStats | None] = []

    augmenter = default_augmenter(word_to_casings, device)

    for feature_index in feature_indices:
        print(f"{feature_index=}", flush=True)
        train_samples, base_max_activation = feature_samples(feature_index)

        train_samples, test_samples = train_test_split(  # type: ignore
            train_samples, train_size=train_config.train_proportion, random_state=train_config.random_seed
        )

        try:
            neuron_model, stats = train_and_eval.train_and_eval(
                feature_activation(feature_index),
                tokenizer,
                augmenter,
                train_samples,
                test_samples,
                base_max_activation,
                fire_threshold=train_config.fire_threshold,
                fit_config=train_config.fit_config,
            )
        except Exception as e:
            if train_config.stop_on_error:
                raise e
            print(f"Error on feature {feature_index}", file=sys.stderr)
            traceback.print_exception(e)

            neuron_model = None
            stats = None

        feature_models.append(neuron_model)
        feature_stats.append(stats)

    return feature_models, feature_stats


def run_training(
    model: HookedTransformer,
    layer_indices: List[int],
    neuron_indices: List[int],
    layer_ending: str,
    augmenter: Augmenter,
    model_name: str,
    neuron_store: NeuronStore,
    all_stats: Dict[int, Dict[int, NeuronStats]],
    config: TrainConfig,
) -> Tuple[Dict[int, Dict[int, NeuronModel]], NeuronStore, Dict[int, Dict[int, NeuronStats]]]:
    random.seed(config.random_seed)

    neuron_models: Dict[int, Dict[int, NeuronModel]] = {}

    tokenizer = Tokenizer(model)

    for layer_index in layer_indices:
        all_stats[layer_index] = {}
        neuron_models[layer_index] = {}
        for neuron_index in neuron_indices:
            print(f"{layer_index=} {neuron_index=}", flush=True)
            samples, base_max_activation = scrape.scrape_neuron(model_name, layer_index, neuron_index)

            train_samples: list[str]
            test_samples: list[str]
            train_samples, test_samples = train_test_split(  # type: ignore
                samples, train_size=config.train_proportion, random_state=config.random_seed
            )

            layer_id = layer_index_to_name(layer_index, layer_ending)

            neuron_model, stats = train_and_eval.train_and_eval(
                feature_activation(model, layer_id, neuron_index),
                tokenizer,
                augmenter,
                train_samples,
                test_samples,
                base_max_activation,
                fire_threshold=config.fire_threshold,
                fit_config=config.fit_config,
            )

            neuron_model.update_neuron_store(neuron_store, str(layer_index), neuron_index)

            neuron_models[layer_index][neuron_index] = neuron_model
            all_stats[layer_index][neuron_index] = stats

    return neuron_models, neuron_store, all_stats


def run(
    model_name: str,
    layer_indices: List[int],
    neuron_indices: List[int],
    layer_ending: str,
    word_to_casings: WordToCasings,
    aug_model_name: str,
    neuron_store: NeuronStore,
    all_stats: Dict[int, Dict[int, NeuronStats]],
    config: TrainConfig,
    device: device,
) -> Tuple[Dict[int, Dict[int, NeuronModel]], NeuronStore, Dict[int, Dict[int, NeuronStats]]]:
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
