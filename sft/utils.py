import torch
from pathlib import Path
import logging
import transformers


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[]):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    return lora_module_names




def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args):
    vision_model_params = model.vision_tower.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.multi_modal_projector.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = model.language_model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)




def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return



def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        trainer.model.config.save_pretrained(output_dir)

def shift_padding_side(
        tokens: torch.Tensor,
        ar_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        targets: torch.Tensor,
        padding_side: str = "right",
    ) -> tuple[torch.Tensor]:
    if padding_side not in ["right", "left"]:
        return tokens, ar_mask, padding_mask, loss_mask, targets

    new_tokens = torch.empty_like(tokens)
    new_ar_masks = torch.empty_like(ar_mask)
    new_padding_mask = torch.empty_like(padding_mask)
    new_loss_mask = torch.empty_like(loss_mask)
    new_targets = torch.empty_like(targets)
    batch_size = tokens.shape[0]
    for i in range(batch_size):
        padding_indices = torch.where(padding_mask[i] == 0)[0]
        non_padding_indices = torch.where(padding_mask[i] == 1)[0]
        if padding_side == "left":
            new_indices = torch.cat((padding_indices, non_padding_indices), dim=0)
        else:
            new_indices = torch.cat((non_padding_indices, padding_indices), dim=0)
        new_tokens[i] = tokens[i].index_select(0, new_indices)
        new_ar_masks[i] = ar_mask[i].index_select(0, new_indices)
        new_padding_mask[i] = padding_mask[i].index_select(0, new_indices)
        new_loss_mask[i] = loss_mask[i].index_select(0, new_indices)
        new_targets[i] = targets[i].index_select(0, new_indices)

    return new_tokens, new_ar_masks, new_padding_mask, new_loss_mask, new_targets

def jax_causal_mask(padding_mask, attention_mask, dtype):
    min_dtype = torch.finfo(dtype).min
    attn_mask = (
        torch.cumsum(attention_mask, dim=1)[:, None, :]
        <= torch.cumsum(attention_mask, dim=1)[:, :, None]
    )
    valid_mask = padding_mask[:, None, :] * padding_mask[:, :, None]
    block_causal_mask = (attn_mask & valid_mask).to(torch.bool)
    causal_mask = torch.where(block_causal_mask, 0.0, min_dtype)

    return causal_mask[:, None, :, :]

