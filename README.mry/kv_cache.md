# CODE

kv_cache定义：

```c
struct model_kv_cache {
  struct ne_tensor* k = NULL;
  struct ne_tensor* v = NULL;
  struct ne_tensor* cossin = NULL;  // cached cos/sin value for shifting RoPE

  struct ne_context* ctx = NULL;

  model_ctx_buffer buf;

  int n;  // number of tokens currently in the cache

  bool has_shift = false;  // ring-buffer (for too long text generation like streaming-llm)
  std::vector<kv_seq_cell> seq_cells;

  ~model_kv_cache() {
    if (ctx) {
      ne_free(ctx);
    }
  }
};
```

量化的过程：

```python
// intel_extension_for_transformers\llm\quantization\utils.py
def convert_to_quantized_model(model, config):
    replace_linear()
```

推理的时候，在这个地方调用了jblas：

```python
// intel_extension_for_transformers\llm\quantization\nn\modules.py: 107 forward()
def matmul_kbit(A: Tensor, B: Tensor, bias, out, compute_dtype, weight_dtype, do_dequant=False):
    if do_dequant:
        return MatMulKBit.apply(A, B, out, bias, compute_dtype, weight_dtype)
    else:
        torch.ops.weight_only_jblasop.qbits_linear(
            A, B.data, bias, out, out.shape[-1], bias is not None, compute_dtype, weight_dtype
        )
        return out
```

# DOC

D:\SJTUwork\PIM-AI\github\intel-extension-for-transformers\intel_extension_for_transformers\llm\runtime\graph\docs

D:\SJTUwork\PIM-AI\github\intel-extension-for-transformers\intel_extension_for_transformers\llm\runtime\graph\core

