import datasets  # type: ignore[missingTypeStubs, import-untyped]
import torch
import transformer_lens  # type: ignore
from datasets import IterableDataset  # type: ignore[missingTypeStubs]
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer  # type: ignore[import]
from transformer_lens.hook_points import HookPoint  # type: ignore[import]


class SparseAutoencoder:
    _w_enc: Float[Tensor, "layer_dim num_features"]
    _b_enc: Float[Tensor, " num_features"]
    _w_dec: Float[Tensor, " num_features layer_dim"]
    _b_dec: Float[Tensor, " layer_dim"]

    def __init__(
        self,
        w_enc: Float[Tensor, "layer_dim num_features"],
        b_enc: Float[Tensor, " num_features"],
        w_dec: Float[Tensor, " num_features layer_dim"],
        b_dec: Float[Tensor, " layer_dim"],
        device: str,
    ) -> None:
        self._w_enc = w_enc.to(device)
        self._b_enc = b_enc.to(device)
        self._w_dec = w_dec.to(device)
        self._b_dec = b_dec.to(device)

    @staticmethod
    def from_data(data: dict[str, torch.Tensor], device: str) -> "SparseAutoencoder":
        return SparseAutoencoder(data["W_enc"], data["b_enc"], data["W_dec"], data["b_dec"], device)

    def encode(self, x: Float[Tensor, "*batch layer_dim"]) -> Float[Tensor, "*batch num_sae_features"]:
        return torch.relu(x @ self._w_enc + self._b_enc)


point = "resid_pre"
layer_index = 6

model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
sae_data = transformer_lens.utils.download_file_from_hf(
    "jacobcd52/gpt2-small-sparse-autoencoders",
    f"gpt2-small_6144_{point}_{layer_index}.pt",
    force_is_torch=True,
)
sae = SparseAutoencoder.from_data(sae_data, "cuda")

dataset: IterableDataset = datasets.load_dataset(  # type: ignore[reportUnknownMemberType]
    "monology/pile-uncopyrighted", streaming=True, split="train", trust_remote_code=True
)

samples_text = [sample["text"] for sample in dataset.take(5)]

sample_tokens: Int[Tensor, "num_samples sample_length"] = model.to_tokens(samples_text)

activations: Float[Tensor, "num_samples sample_length num_sae_features"] = torch.full(
    sample_tokens.shape + (6144,), float("nan")
)


def hook(activation: Float[Tensor, "num_samples sample_length layer_dim"], hook: HookPoint) -> None:
    activations[:] = sae.encode(activation)


layer_id = f"blocks.{layer_index}.hook_resid_pre"

with torch.no_grad():
    model.run_with_hooks(sample_tokens, fwd_hooks=[(layer_id, hook)])
    assert not torch.isnan(activations).any(), "Activations should not contain NaNs"

print(activations.count_nonzero((0, 1)).topk(10))
print(activations.sum((0, 1)).topk(10))
