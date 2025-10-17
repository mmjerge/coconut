# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import DynamicCache

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


class Coconut(nn.Module):
    """
    Coconut: Chain of Continuous Thought model.

    Implements continuous latent reasoning by replacing special <|latent|> tokens
    with continuous hidden state vectors that are fed back iteratively through
    the model for multi-step reasoning.

    Attributes:
        base_causallm: The underlying causal language model (e.g., GPT2, Llama3)
        latent_token_id: Token ID for <|latent|> (continuous thought placeholder)
        start_latent_id: Token ID for <|start-latent|> marker
        end_latent_id: Token ID for <|end-latent|> marker
        eos_token_id: Token ID for end-of-sequence
        embedding: The input embedding layer from the base model
        gen_forward_cnt: Counter for number of forward passes during generation
    """

    def __init__(
        self,
        base_causallm: nn.Module,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
    ) -> None:
        """
        Initialize the Coconut model wrapper.

        Args:
            base_causallm: Pre-trained causal language model to wrap
            latent_token_id: Token ID for <|latent|> tokens
            start_latent_id: Token ID for <|start-latent|> marker
            end_latent_id: Token ID for <|end-latent|> marker
            eos_token_id: Token ID for end-of-sequence
        """
        super(Coconut, self).__init__()
        self.gen_forward_cnt: int = 0
        self.base_causallm: nn.Module = base_causallm
        self.latent_token_id: int = latent_token_id
        self.eos_token_id: int = eos_token_id
        self.start_latent_id: int = start_latent_id
        self.end_latent_id: int = end_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding: nn.Embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding: nn.Embedding = self.base_causallm.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        position_ids: torch.Tensor,
        noise_scale: float = 0.0,
        **kwargs
    ) -> Outputs:
        """
        Forward pass implementing continuous latent reasoning.

        TRUE METHOD: When <start-latent> and <end-latent> are adjacent (or only contain
        explicit <latent> tokens), the model automatically performs MAX_N_LATENT (8) 
        latent reasoning passes in continuous hidden state space.

        Args:
            input_ids: Token IDs. Shape: (batch_size, seq_len)
            attention_mask: Attention mask for input. Shape: (batch_size, seq_len)
            labels: Target token IDs for loss computation. Shape: (batch_size, seq_len)
            position_ids: Position IDs for each token. Shape: (batch_size, seq_len)
            noise_scale: Scale of Gaussian noise to add to first pass hidden states. Default: 0.0
            **kwargs: Additional arguments (unused)

        Returns:
            Outputs: Named tuple with:
                - loss: Cross-entropy loss (scalar tensor)
                - inputs_embeds: Final input embeddings. Shape: (batch_size, seq_len, hidden_size)
                - logits: Output logits. Shape: (batch_size, seq_len, vocab_size)
        """
        logits: List[torch.Tensor] = []
        
        # Get device from input
        device = input_ids.device

        # Step 1: Detect latent reasoning mode
        latent_mask: torch.Tensor = input_ids == self.latent_token_id
        latent_indices: torch.Tensor = latent_mask.nonzero()
        
        start_latent_mask: torch.Tensor = input_ids == self.start_latent_id
        end_latent_mask: torch.Tensor = input_ids == self.end_latent_id
        start_latent_indices: torch.Tensor = start_latent_mask.nonzero()
        end_latent_indices: torch.Tensor = end_latent_mask.nonzero()
        
        has_explicit_latent_tokens = len(latent_indices) > 0
        has_latent_markers = len(start_latent_indices) > 0 and len(end_latent_indices) > 0
        
        if has_latent_markers and not has_explicit_latent_tokens:
            # TRUE METHOD
            print(f"TRUE METHOD: Detected <start-latent> and <end-latent> markers")
            print(f"Automatically performing {MAX_N_LATENT} latent reasoning passes")
            
            batch_size = input_ids.shape[0]
            latent_lists: List[List[int]] = []
            
            for batch_idx in range(batch_size):
                batch_start_indices = [idx[1].item() for idx in start_latent_indices if idx[0] == batch_idx]
                batch_end_indices = [idx[1].item() for idx in end_latent_indices if idx[0] == batch_idx]
                
                if len(batch_start_indices) > 0 and len(batch_end_indices) > 0:
                    start_pos = batch_start_indices[0]
                    virtual_latent_positions = [start_pos + i + 1 for i in range(MAX_N_LATENT)]
                    latent_lists.append(virtual_latent_positions)
                else:
                    latent_lists.append([])
            
            expanded_input_ids = []
            expanded_attention_mask = []
            expanded_position_ids = []
            expanded_labels = []
            
            for batch_idx in range(batch_size):
                batch_start_indices = [idx[1].item() for idx in start_latent_indices if idx[0] == batch_idx]
                
                if len(batch_start_indices) > 0:
                    start_pos = batch_start_indices[0]
                    
                    before_start = input_ids[batch_idx, :start_pos + 1]
                    after_start = input_ids[batch_idx, start_pos + 1:]
                    
                    virtual_latents = torch.full((MAX_N_LATENT,), self.latent_token_id, 
                                                dtype=input_ids.dtype, device=device)
                    
                    expanded_input_ids.append(torch.cat([before_start, virtual_latents, after_start]))
                    
                    before_mask = attention_mask[batch_idx, :start_pos + 1]
                    after_mask = attention_mask[batch_idx, start_pos + 1:]
                    virtual_mask = torch.ones(MAX_N_LATENT, dtype=attention_mask.dtype, device=device)
                    expanded_attention_mask.append(torch.cat([before_mask, virtual_mask, after_mask]))
                    
                    before_pos = position_ids[batch_idx, :start_pos + 1]
                    after_pos = position_ids[batch_idx, start_pos + 1:] + MAX_N_LATENT
                    virtual_pos = torch.arange(start_pos + 1, start_pos + 1 + MAX_N_LATENT, 
                                            dtype=position_ids.dtype, device=device)
                    expanded_position_ids.append(torch.cat([before_pos, virtual_pos, after_pos]))
                    
                    before_labels = labels[batch_idx, :start_pos + 1]
                    after_labels = labels[batch_idx, start_pos + 1:]
                    virtual_labels = torch.full((MAX_N_LATENT,), -100, dtype=labels.dtype, device=device)
                    expanded_labels.append(torch.cat([before_labels, virtual_labels, after_labels]))
                else:
                    expanded_input_ids.append(input_ids[batch_idx])
                    expanded_attention_mask.append(attention_mask[batch_idx])
                    expanded_position_ids.append(position_ids[batch_idx])
                    expanded_labels.append(labels[batch_idx])
            
            input_ids = torch.stack(expanded_input_ids)
            attention_mask = torch.stack(expanded_attention_mask)
            position_ids = torch.stack(expanded_position_ids)
            labels = torch.stack(expanded_labels)
            
            latent_mask = input_ids == self.latent_token_id
            latent_indices = latent_mask.nonzero()
        
        # Step 2: Reorganize latent indices by batch
        latent_lists: List[List[int]] = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents: int = max([len(l) for l in latent_lists]) if len(latent_lists) > 0 and any(latent_lists) else 0
        print(f"Max latent tokens to process: {max_n_latents}")

        # Step 3: Initialize
        next_compute_range: Tuple[int, int] = (0, input_ids.shape[1])
        inputs_embeds: torch.Tensor = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache: Optional[Tuple] = None

        # Step 4: Iterative reasoning
        for pass_idx in range(max_n_latents):
            print(f"  Latent pass {pass_idx + 1}/{max_n_latents}")

            if kv_cache == None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset: int = 0
            else:
                # Handle Cache API
                if isinstance(kv_cache, tuple):
                    cache_obj = DynamicCache()
                    for layer_idx, (k, v) in enumerate(kv_cache):
                        k_truncated = k[:, :, : next_compute_range[0], :]
                        v_truncated = v[:, :, : next_compute_range[0], :]
                        cache_obj.update(k_truncated, v_truncated, layer_idx)
                    past_key_values = cache_obj
                elif kv_cache is not None:
                    truncated_cache = DynamicCache()
                    for layer_idx in range(len(kv_cache)):
                        k, v = kv_cache[layer_idx]
                        k_truncated = k[:, :, : next_compute_range[0], :]
                        v_truncated = v[:, :, : next_compute_range[0], :]
                        truncated_cache.update(k_truncated, v_truncated, layer_idx)
                    past_key_values = truncated_cache
                else:
                    past_key_values = None

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset: int = next_compute_range[0]

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1),
            )

            hidden_states: torch.Tensor = outputs.hidden_states[-1]
            
            if pass_idx == 0 and noise_scale > 0.0:
                noise = torch.randn_like(hidden_states) * noise_scale
                hidden_states = hidden_states + noise
                print(f"  Applied noise with scale {noise_scale} to first pass hidden states")
            
            kv_cache = outputs.past_key_values

            # Step 5: Replace latent embeddings
            filling_indices: List[Tuple[int, int]] = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            tensor_list: List[List[torch.Tensor]] = [
                [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            inputs_embeds = torch.stack([
                torch.stack(tensor_list[batch_idx])
                for batch_idx in range(inputs_embeds.shape[0])
            ])

        # Step 6: Final forward pass
        if kv_cache is not None:
            if isinstance(kv_cache, tuple):
                final_cache = DynamicCache()
                for layer_idx, (k, v) in enumerate(kv_cache):
                    k_truncated = k[:, :, : next_compute_range[0], :]
                    v_truncated = v[:, :, : next_compute_range[0], :]
                    final_cache.update(k_truncated, v_truncated, layer_idx)
                past_key_values_final = final_cache
            else:
                truncated_cache = DynamicCache()
                for layer_idx in range(len(kv_cache)):
                    k, v = kv_cache[layer_idx]
                    k_truncated = k[:, :, : next_compute_range[0], :]
                    v_truncated = v[:, :, : next_compute_range[0], :]
                    truncated_cache.update(k_truncated, v_truncated, layer_idx)
                past_key_values_final = truncated_cache
        else:
            past_key_values_final = None

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=past_key_values_final,
            output_hidden_states=True,
        )

        logits.append(outputs.logits)
        self.gen_forward_cnt += max_n_latents + 1
        print(f"Total forward passes: {self.gen_forward_cnt}")

        # Step 7: Compute loss
        logits = torch.cat(logits, dim=-2)
        shift_logits: torch.Tensor = logits[..., :-1, :].contiguous()
        shift_labels: torch.Tensor = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        loss: torch.Tensor = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self) -> None:
        """Set the model to training mode."""
        self.base_causallm.train()

    def eval(self) -> None:
        """Set the model to evaluation mode."""
        self.base_causallm.eval()

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 16,
        output_embedding: bool = False,
        synced_gpus: bool = False,
        noise_scale: float = 0.0,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate text autoregressively with continuous latent reasoning.

        Processes the input through continuous latent reasoning (with optional noise),
        then generates new tokens one at a time using greedy decoding (argmax).

        Args:
            input_ids: Input token IDs including <|latent|> tokens. Shape: (1, seq_len)
            attention_mask: Attention mask (unused in current implementation)
            max_new_tokens: Maximum number of new tokens to generate
            output_embedding: If True, return both tokens and final embeddings
            synced_gpus: If True, sync forward passes across GPUs (for FSDP)
            noise_scale: Scale of Gaussian noise to add to first latent pass. Default: 0.0
            **kwargs: Additional arguments (unused)

        Returns:
            If output_embedding=False: Generated token IDs. Shape: (1, total_seq_len)
            If output_embedding=True: Tuple of (token_ids, final_embeddings)

        Note:
            Currently only supports batch_size=1 during generation.
        """

        # Reset forward pass counter
        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        # Track generated token IDs
        tokens: List[int] = input_ids[0].detach().tolist()

        # Run forward pass with continuous latent reasoning (WITH NOISE)
        labels: torch.Tensor = input_ids.clone()  # placeholder, not used
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
            noise_scale=noise_scale  # CRITICAL: Pass noise to forward!
        )
        inputs_embeds: torch.Tensor = outputs.inputs_embeds

        # Generate first token using greedy decoding (argmax)
        next_token: int = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed: torch.Tensor = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds: torch.Tensor = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # Generate remaining tokens autoregressively
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()

            # Stop if end-of-sequence token is generated
            if next_token == self.eos_token_id:
                break

            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # Synchronize forward pass count across devices for FSDP
            # All devices must perform the same number of forward passes
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        # Return results based on output_embedding flag
        if output_embedding:
            # For analysis: return both tokens and final embeddings
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)