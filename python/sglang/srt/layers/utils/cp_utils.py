"""Import-only shims for deprecated platform backends awaiting CP refactoring.

The legacy CP algorithms have been removed. These names keep the retained
NPU/MUSA attention backends importable for non-CP inference; calling them fails.
"""


def _deprecated_platform_cp():
    raise ValueError(
        "Prefill CP on HIP/NPU/MUSA is deprecated; CP support will be refactored soon."
    )


def mla_prefill_cp_use_mha_one_shot(forward_batch, backend_name: str) -> bool:
    """Whether this CP prefill step can run the MHA one-shot formulation.

    Absorbed MLA attends in the (kv_lora_rank + rope) latent space, which
    costs ~3.4x the flops per query-key pair of the MHA formulation. Under the
    contiguous strategy queries are rank-local and the latent pool is fully
    populated by materialize_full_mla_kv, so each rank can up-project its causal
    window and run the same one-shot varlen kernel; zigzag keeps absorbed MLA.
    """
    from sglang.srt.layers.cp.base import ContextParallelStrategyKind, get_cp_strategy
    from sglang.srt.layers.cp.utils import is_cp_active

    if backend_name != "fa3":
        return False
    if not is_cp_active(forward_batch):
        return False
    strategy = get_cp_strategy()
    if strategy is None or strategy.kind != ContextParallelStrategyKind.CONTIGUOUS:
        return False
    meta = forward_batch.attn_cp_metadata
    if not meta.kv_len_prev_list:
        return False
    # Bound by the full (prefix + extend) window so every rank makes the same choice
    if forward_batch.seq_lens_cpu is None:
        return False
    return sum(forward_batch.seq_lens_cpu) <= forward_batch.get_max_chunk_capacity()


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    _deprecated_platform_cp()


def cp_all_gather_rerange_kv_cache(input_tensor, cp_size, forward_batch, stream):
    _deprecated_platform_cp()


def cp_allgather_and_save_kv_cache(forward_batch, layer, k, v, cp_size, swa_loc=None):
    _deprecated_platform_cp()
