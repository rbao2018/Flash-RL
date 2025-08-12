# ⚡ FlashRL Verification Tutorial ⚡

This tutorial provides methods to verify that FlashRL is correctly installed and functioning in your environment.

## Prerequisites

### Initial Setup
```bash 
git clone --depth 1 --branch v0.4.8 https://github.com/EleutherAI/lm-evaluation-harness && pip install -e lm-evaluation-harness
```

## FP8 Quantization Verification

This section contains two tests to verify that both core FlashRL patches are working correctly.

### Test 1: Verifying `LLM.__init__` Patch

**Purpose**: This test verifies that FlashRL can successfully override vLLM's initialization process.

**How it works**: FlashRL forces vLLM to load `LiyuanLucasLiu/Qwen2.5-0.5B-Instruct-VERL` in FP8 quantization instead of `Qwen/Qwen2.5-0.5B-Instruct` in BF16. Since the VERL model has slightly better performance than the base model, successful patching should show improved results.

**Expected Results**:
- **Target Performance**: ~0.36 (flexible-extract), meaning the patch succeeds
- **Alternative Performance**: ~0.34 (flexible-extract), meaning the patch fails

**Commands**:
```bash 
flashrl setup --fn fp8 --model LiyuanLucasLiu/Qwen2.5-0.5B-Instruct-VERL -o $HOME/.flashrl_config.yaml

# Run lm-eval with FlashRL overriding, using FP8 quantization
FLASHRL_CONFIG=$HOME/.flashrl_config.yaml \
FLASHRL_LOGGING_LEVEL=DEBUG \
VLLM_LOGGING_LEVEL=DEBUG \
CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes=1 --nproc-per-node=1 --no-python lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks gsm8k \
    --batch_size auto
```

**Expected Output**:
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.3632|±  |0.0132|

### Test 2: Verifying `load_weights` Patch

**Purpose**: This test verifies that FlashRL can successfully handle multiple weight loading operations.

**How it works**: FlashRL instructs vLLM to call `load_weights` twice at the end of its initialization:
1. First load: `Qwen/Qwen2.5-0.5B-Instruct` (baseline model)
2. Second load: `LiyuanLucasLiu/Qwen2.5-0.5B-Instruct-VERL` (improved model)

Both models are loaded in FP8 quantization. Since the VERL model performs better, seeing the improved performance indicates successful patching.

**Expected Results**:
- **Target Performance**: ~0.36 (flexible-extract), meaning the patch succeeds
- **Alternative Performance**: ~0.34 (flexible-extract), meaning the patch fails

**Commands**:
```bash 
# Run lm-eval with FlashRL overriding, testing dual weight loading with FP8 quantization
FLASHRL_TEST_RELOAD="Qwen/Qwen2.5-0.5B-Instruct,LiyuanLucasLiu/Qwen2.5-0.5B-Instruct-VERL" \
FLASHRL_CONFIG="fp8" \
FLASHRL_LOGGING_LEVEL=DEBUG \
VLLM_LOGGING_LEVEL=DEBUG \
CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes=1 --nproc-per-node=1 --no-python lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks gsm8k \
    --batch_size auto
```

**Expected Output**:
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.3632|±  |0.0132|

## Interpreting Results

### Success Indicators
- **Performance ~0.36**: FlashRL patches are working correctly
- **Performance ~0.34**: FlashRL patches may not be functioning; the system is likely using the baseline model

### Troubleshooting
If you don't see the expected performance improvements:

1. **Check Installation**: Verify FlashRL is properly installed
2. **Review Logs**: Examine DEBUG logs for error messages
