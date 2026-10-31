# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""mHC layer communicator for hybrid linear/DSA models under prefill CP.

A rank keeps its own tokens through the whole decoder layer, as with the DSA-CP
communicator, and gathers across the CP group only around a full-TP MLP; the
hyper-connection streams are split and combined locally on either side of it.
"""

from functools import partial

import torch

from sglang.kernels.ops.layernorm.mhc import hc_contract
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.communicator import CommunicateContext, ScatterMode
from sglang.srt.layers.communicator_dsa_cp import (
    DSACPCommunicateSimpleFn,
    DSACPCommunicateSummableTensorPairFn,
    DSACPCommunicateWithAllReduceAndLayerNormFn,
    dsa_cp_gather_hidden_states,
    dsa_cp_reduce_scatter_hidden_states,
)
from sglang.srt.layers.communicator_mhc import MHCLayerCommunicator, MHCState
from sglang.srt.layers.cp.utils import is_mla_cp_active
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel


def _is_cp_sharded(forward_batch: ForwardBatch) -> bool:
    return dsa_use_prefill_cp(forward_batch) or is_mla_cp_active(forward_batch)


class MHCHybridDSACPLayerCommunicator(MHCLayerCommunicator):
    def _post_init_communicate(self):
        # The hyper-connection mixes are not split across attn-TP ranks here
        if get_parallel().attn_tp_size > 1:
            raise NotImplementedError(
                "mHC prefill context parallelism requires attention TP size 1"
            )
        if self.layer_scatter_modes.mlp_mode != ScatterMode.SCATTERED:
            if self._context.attn_dp_size != 1:
                raise NotImplementedError(
                    "mHC prefill context parallel without an MoE runner backend "
                    f"needs attn_dp_size 1, got {self._context.attn_dp_size}; drop "
                    "--enable-dp-attention or set --moe-runner-backend"
                )
        self._communicate_simple_fn = DSACPCommunicateSimpleFn.get_fn(
            input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = (
            MHCHybridDSACPCommunicateWithAllReduceAndLayerNormFn.get_fn(
                hidden_states_input_mode=ScatterMode.SCATTERED,
                residual_input_mode=ScatterMode.SCATTERED,
                hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,
                residual_output_mode=ScatterMode.SCATTERED,
                context=self._context,
            )
        )
        self._communicate_summable_tensor_pair_fn = (
            MHCHybridDSACPCommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
                residual_input_mode=ScatterMode.SCATTERED,
                output_mode=ScatterMode.SCATTERED,
                context=self._context,
            )
        )

    def should_use_reduce_scatter(self, forward_batch: ForwardBatch):
        # The CP scatter reduces the full-TP MLP output itself
        return (
            self._communicate_summable_tensor_pair_fn
            is MHCHybridDSACPCommunicateSummableTensorPairFn._scatter_hidden_states
            and _is_cp_sharded(forward_batch)
        )


class MHCHybridDSACPCommunicateWithAllReduceAndLayerNormFn(
    DSACPCommunicateWithAllReduceAndLayerNormFn
):
    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        hidden_states_output_mode: ScatterMode,
        residual_output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        for name, mode in (
            ("hidden_states_input_mode", hidden_states_input_mode),
            ("residual_input_mode", residual_input_mode),
            ("residual_output_mode", residual_output_mode),
        ):
            if mode != ScatterMode.SCATTERED:
                raise NotImplementedError(
                    f"mHC prefill context parallel needs {name} SCATTERED, got {mode}"
                )
        if hidden_states_output_mode == ScatterMode.SCATTERED:
            return MHCHybridDSACPCommunicateWithAllReduceAndLayerNormFn._simple
        if hidden_states_output_mode == ScatterMode.FULL:
            return partial(
                MHCHybridDSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,
            )
        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} "
            f"{hidden_states_output_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _simple(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        mhc: MHCState,
    ):
        return mhc.attn_to_mlp(hidden_states, residual, out_norm=layernorm)

    @staticmethod
    def _gather_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
        mhc: MHCState,
    ):
        hidden_states, residual = mhc.attn_to_mlp(
            hidden_states, residual, out_norm=layernorm
        )
        # The residual streams stay rank-local; only the MLP input is gathered
        if _is_cp_sharded(forward_batch):
            hidden_states = dsa_cp_gather_hidden_states(hidden_states)
        return hidden_states, residual


class MHCHybridDSACPCommunicateSummableTensorPairFn(
    DSACPCommunicateSummableTensorPairFn
):
    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if (
            hidden_states_input_mode == ScatterMode.FULL
            and residual_input_mode == ScatterMode.SCATTERED
            and output_mode == ScatterMode.SCATTERED
        ):
            return MHCHybridDSACPCommunicateSummableTensorPairFn._scatter_hidden_states
        if context.is_same_group_size(
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):
            return MHCHybridDSACPCommunicateSummableTensorPairFn._trivial
        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    @staticmethod
    def _combine(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        mhc: MHCState,
        is_last_layer: bool,
    ):
        hidden_states = mhc.mlp_combine(hidden_states, residual)
        if is_last_layer:
            hidden_states = hc_contract(hidden_states, mhc.hc_mult)
        return hidden_states, None

    @staticmethod
    def _trivial(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        *,
        mhc: MHCState,
        is_last_layer: bool,
        **kwargs,
    ):
        return MHCHybridDSACPCommunicateSummableTensorPairFn._combine(
            hidden_states, residual, mhc, is_last_layer
        )

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        allow_reduce_scatter: bool = False,
        *,
        mhc: MHCState,
        is_last_layer: bool,
        **kwargs,
    ):
        # Full-TP MLP output -> this rank's tokens, reduced across the CP group
        if _is_cp_sharded(forward_batch):
            hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)
        return MHCHybridDSACPCommunicateSummableTensorPairFn._combine(
            hidden_states, residual, mhc, is_last_layer
        )
