# Copyright 2026 Google LLC
#
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

import torch
from torch.nn import Parameter
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_torchtpu.layers.adapter import pallas_tp_all_reduce


@RowParallelLinear.register_oot
class TpuRowParallelLinear(RowParallelLinear):
    """RowParallelLinear that adds the bias after the TP all-reduce.

    Upstream fuses the bias into the GEMM on rank 0 only, so the per-rank FX
    graphs differ and the torch_tpu XLA backend hands the all_reduce mismatched
    channel IDs, breaking the collective. The bias is replicated on every rank,
    so adding it afterwards sums the same terms in a different order and keeps
    the graphs identical.

    Only the deferred case and a layer armed by pallas_tp_all_reduce
    (TPU_PALLAS_TP_ALL_REDUCE) are overridden; everything else delegates to
    ``super().forward()``.
    """

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if pallas_tp_all_reduce.armed_op(self) is not None:
            return self._forward_pallas_all_reduce(input_)
        defer_bias = (
            self.reduce_results
            and self.tp_size > 1
            and not self.skip_bias_add
            and self.bias is not None
        )
        if not defer_bias:
            return super().forward(input_)

        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        output_parallel = self.quant_method.apply(self, input_parallel, None)
        output = tensor_model_parallel_all_reduce(output_parallel)
        output = output + self.bias

        # skip_bias_add is False here, so there is no bias left to return.
        if not self.return_bias:
            return output
        return output, None

    def _forward_pallas_all_reduce(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        """Upstream's forward with the TP all-reduce through the armed op (the
        bias, if any, deferred past it as above). Arming requires
        reduce_results and tp_size > 1."""
        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        output_parallel = self.quant_method.apply(self, input_parallel, None)
        output = pallas_tp_all_reduce.all_reduce_tp(self, output_parallel)
        if output is None:
            output = tensor_model_parallel_all_reduce(output_parallel)
        if self.bias is not None and not self.skip_bias_add:
            output = output + self.bias

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
