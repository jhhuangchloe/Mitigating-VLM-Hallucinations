import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from typing import Any, Optional, Tuple


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def llama_new_forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.LongTensor] = None,
      past_key_value: Optional[Any] = None,
      output_attentions: bool = False,
      use_cache: bool = False,
      cache_position: Optional[torch.LongTensor] = None,
      position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
      **kwargs,
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
      bsz, q_len, _ = hidden_states.size()

      if self.config.pretraining_tp > 1:
          key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
          query_slices = self.q_proj.weight.split(
              (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
          )
          key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
          value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

          query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
          query_states = torch.cat(query_states, dim=-1)

          key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
          key_states = torch.cat(key_states, dim=-1)

          value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
          value_states = torch.cat(value_states, dim=-1)

      else:
          query_states = self.q_proj(hidden_states)
          key_states = self.k_proj(hidden_states)
          value_states = self.v_proj(hidden_states)

      query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
      key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
      value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

      if position_embeddings is None:
          logger.warning_once(
              "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
              "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
              "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
              "removed and `position_embeddings` will be mandatory."
          )
          cos, sin = self.rotary_emb(value_states, position_ids)
      else:
          cos, sin = position_embeddings
      query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

      if past_key_value is not None:
          # sin and cos are specific to RoPE models; cache_position needed for the static cache
          cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
          key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

      key_states = repeat_kv(key_states, self.num_key_value_groups)
      value_states = repeat_kv(value_states, self.num_key_value_groups)
      attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

      if attention_mask is not None:  # no matter the length, we just slice it
          causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
          attn_weights = attn_weights + causal_mask

      # JUNJUN's modification
      if hasattr(self, "use_attn"):
          use_attn = self.use_attn
          img_start_idx = self.img_start_idx
          img_end_idx = self.img_end_idx
      else:
          use_attn = False

      if hasattr(self, "use_cfg"):
          use_cfg = self.use_cfg
      else:
          use_cfg = False

      if use_attn and not use_cfg:
          attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
              attn_weights[:, :, -1, img_start_idx:img_end_idx].abs() * self.alpha
              + attn_weights[:, :, -1, img_start_idx:img_end_idx]
          )
      # JUNJUN's modification

      # upcast attention to fp32
      attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
      attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
      attn_output = torch.matmul(attn_weights, value_states)

      if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
          raise ValueError(
              f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
              f" {attn_output.size()}"
          )

      attn_output = attn_output.transpose(1, 2).contiguous()

      attn_output = attn_output.reshape(bsz, q_len, -1)

      if self.config.pretraining_tp > 1:
          attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
          o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
          attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
      else:
          attn_output = self.o_proj(attn_output)

      if not output_attentions:
          attn_weights = None

      return attn_output, attn_weights, past_key_value


def llama_forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.LongTensor] = None,
      past_key_value: Optional[Any] = None,
      output_attentions: bool = False,
      use_cache: bool = False,
      cache_position: Optional[torch.LongTensor] = None,
      position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
      **kwargs,
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
      bsz, q_len, _ = hidden_states.size()

      if self.config.pretraining_tp > 1:
          key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
          query_slices = self.q_proj.weight.split(
              (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
          )
          key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
          value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

          query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
          query_states = torch.cat(query_states, dim=-1)

          key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
          key_states = torch.cat(key_states, dim=-1)

          value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
          value_states = torch.cat(value_states, dim=-1)

      else:
          query_states = self.q_proj(hidden_states)
          key_states = self.k_proj(hidden_states)
          value_states = self.v_proj(hidden_states)

      query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
      key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
      value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

      if position_embeddings is None:
          logger.warning_once(
              "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
              "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
              "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
              "removed and `position_embeddings` will be mandatory."
          )
          cos, sin = self.rotary_emb(value_states, position_ids)
      else:
          cos, sin = position_embeddings
      query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

      if past_key_value is not None:
          # sin and cos are specific to RoPE models; cache_position needed for the static cache
          cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
          key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

      key_states = repeat_kv(key_states, self.num_key_value_groups)
      value_states = repeat_kv(value_states, self.num_key_value_groups)
      attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

      if attention_mask is not None:  # no matter the length, we just slice it
          causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
          attn_weights = attn_weights + causal_mask

      # upcast attention to fp32
      attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
      attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
      attn_output = torch.matmul(attn_weights, value_states)

      if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
          raise ValueError(
              f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
              f" {attn_output.size()}"
          )

      attn_output = attn_output.transpose(1, 2).contiguous()

      attn_output = attn_output.reshape(bsz, q_len, -1)

      if self.config.pretraining_tp > 1:
          attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
          o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
          attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
      else:
          attn_output = self.o_proj(attn_output)

      if not output_attentions:
          attn_weights = None

      return attn_output, attn_weights, past_key_value

def llama_attn_forward(
  self,
  hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  position_ids: Optional[torch.LongTensor] = None,
  past_key_value: Optional[Any] = None,
  output_attentions: bool = False,
  use_cache: bool = False,
  cache_position: Optional[torch.LongTensor] = None,
  position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
  **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
  bsz, q_len, _ = hidden_states.size()

  query_states = self.q_proj(hidden_states)
  key_states = self.k_proj(hidden_states)
  value_states = self.v_proj(hidden_states)

  # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
  query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
  key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
  value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

  if position_embeddings is None:
      logger.warning_once(
          "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
          "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
          "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
          "removed and `position_embeddings` will be mandatory."
      )
      cos, sin = self.rotary_emb(value_states, position_ids)
  else:
      cos, sin = position_embeddings
  query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

  if past_key_value is not None:
      # sin and cos are specific to RoPE models; cache_position needed for the static cache
      cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
      key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

  key_states = repeat_kv(key_states, self.num_key_value_groups)
  value_states = repeat_kv(value_states, self.num_key_value_groups)
  attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

  if attention_mask is not None:  # no matter the length, we just slice it
      causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
      attn_weights = attn_weights + causal_mask

  # upcast attention to fp32
  attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
  attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
  attn_output = torch.matmul(attn_weights, value_states)

  if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
      raise ValueError(
          f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
          f" {attn_output.size()}"
      )

  attn_output = attn_output.transpose(1, 2).contiguous()

  attn_output = attn_output.reshape(bsz, q_len, -1)

  attn_output = self.o_proj(attn_output)

  if not output_attentions:
      attn_weights = None

  return attn_output, attn_weights, past_key_value
