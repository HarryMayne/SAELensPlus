from typing import Any, cast

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule
from transformers import AutoModelForCausalLM
from peft import PeftModel


def load_model(
    model_class_name: str,
    model_name: str,
    finetune_checkpoint: str | None = None,
    device: str | torch.device | None = None,
    model_from_pretrained_kwargs: dict[str, Any] | None = None,
) -> HookedRootModule:
    model_from_pretrained_kwargs = model_from_pretrained_kwargs or {}

    if model_class_name == "HookedTransformer":

        # Load model normally
        model = HookedTransformer.from_pretrained(
            model_name=model_name, device=device, **model_from_pretrained_kwargs
        )

        # Load finetune checkpoint if provided
        if finetune_checkpoint is not None:

            # load the finetune from a directory (probably local) and load the base model
            finetune = AutoModelForCausalLM.from_pretrained(finetune_checkpoint)
            base_model = AutoModelForCausalLM.from_pretrained(finetune.name_or_path)

            # Merge the weights
            finetune_merged = PeftModel.from_pretrained(base_model, finetune_checkpoint)
            finetune_merged = finetune_merged.merge_and_unload()
            finetune_merged.eval()
            
            # swap the state dict

        return model
    
    elif model_class_name == "HookedMamba":
        try:
            from mamba_lens import HookedMamba
        except ImportError:
            raise ValueError(
                "mamba-lens must be installed to work with mamba models. This can be added with `pip install sae-lens[mamba]`"
            )
        # HookedMamba has incorrect typing information, so we need to cast the type here
        return cast(
            HookedRootModule,
            HookedMamba.from_pretrained(
                model_name, device=cast(Any, device), **model_from_pretrained_kwargs
            ),
        )
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")
