# Experimental SM89 / Ada optimizations

This fork carries two small ExLlamaV3 1.5.0 specializations for NVIDIA Ada SM89 GPUs.

They were developed and validated on an **RTX 4060 Ti 16 GB** with a dense
**Qwen3.8-27B EXL3 3.0 bpw** workload. Other SM89 GPUs and other model shapes have
not yet been characterized to the same extent, so treat these changes as experimental.

Upstream base commit:

`02aef45cd681b960a00afcd0749a4ab99e6c1bfe`

## Changes

### 1. SM89 F16ACC GEMM layout

When the existing F16ACC eligibility and rate-probe logic selects the F16ACC path,
SM89 uses the tuned `128x64x64` layout:

- `BM=128`
- `BN=64`
- `BK=64`
- 4 warps (`2x2`)
- `GROUP_M=16`
- `PAD=0`

All non-SM89 architectures keep the upstream dispatch behavior.

The real-shape study found that a more complex per-shape dispatcher was not justified:
the best global configuration captured nearly all of the measured oracle opportunity.

Measured 32K server A/B:

| Configuration | Prefill |
|---|---:|
| ExLlamaV3 1.5.0 control | 652.37 tok/s |
| SM89 F16ACC candidate | 679.31 tok/s |
| Uplift | **+4.129%** |

Additional measured prefill uplift:

- 16K: **+4.71%**
- 64K: **+3.52%**

Correctness gate:

- 16/16 real F16ACC shapes passed
- 3 deterministic seeds per shape
- `max_abs=0`
- no NaN/Inf

### 2. SM89 long-query paged attention

For the cache-backed staged/fp16 long-query path with `hd_pad <= 256`, SM89 uses:

- `BLOCK_M=64`
- `BLOCK_N=32`
- `num_warps=4`
- `num_stages=2`

The upstream control for this tested path used `BM64/BN32/W8/S2`, so the effective
specialization is a reduction from 8 warps to 4 warps. The direct packed
quantized-cache path is intentionally left on the upstream configuration.

Server A/B with the SM89 F16ACC specialization enabled in both arms:

| Context | Control | SM89 attention | Uplift |
|---:|---:|---:|---:|
| 16K | 763.530 tok/s | 787.156 tok/s | **+3.094%** |
| 32K | 680.912 tok/s | 723.184 tok/s | **+6.208%** |
| 64K | 555.397 tok/s | 613.711 tok/s | **+10.500%** |

The 32K result slightly exceeded the prior projection of about 721.6 tok/s.

Correctness and stability during the formal A/B:

- 16K / 32K / 64K: `max_abs=0`
- no VRAM increase in the measured runs
- 14,488 MiB peak at 16K/32K
- 14,608 MiB peak at 64K
- no OOM
- no CUDA illegal-access error
- no Xid
- no server crash

## Combined observed effect

Using the matched 32K measurements above as a directional summary, the path moved from
about **652 tok/s** on the original ExLlamaV3 1.5.0 control to about **723 tok/s** with
both SM89 specializations enabled.

Do not interpret this as a universal ExLlamaV3 speedup. The result depends on the GPU,
model shapes, cache configuration, context length and benchmark methodology.

## Test configuration

The validation workload used:

- NVIDIA RTX 4060 Ti 16 GB, SM89
- ExLlamaV3 1.5.0
- Qwen3.8-27B EXL3 3.0 bpw
- int4 KV cache
- `EXL3_QC_STAGING=1`
- chunk size 4096
- MTP enabled
- vision disabled during performance tests
- cold requests with `cached_tokens=0`

## Important limitation

After the formal paged-attention A/B had completed and the server was shut down, the
test host later lost its `/dev/nvidia*` device nodes before a planned 32K semantic
smoke test could run. The formal benchmark itself had completed without Xid, CUDA
errors or a GPU reset. No causal claim is made here about that later host/driver event.

Because of that unresolved host incident, this branch should remain **experimental**
until it has seen more testing on additional SM89 systems and workloads.

## Using the branch

```sh
git clone https://github.com/grimlee/exllamav3.git
cd exllamav3
git checkout sm89-ada-optimizations
```

Build from source using the normal ExLlamaV3 instructions.

## Scope

This branch intentionally does **not**:

- add a shape-aware F16ACC dispatcher
- change non-SM89 GEMM dispatch
- tune decode attention
- change the direct packed quantized-cache attention path
- change quantization formats or model files

The goal is to keep the tested SM89 changes small, inspectable and easy to revert.
