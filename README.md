# Medical AI Performance Analysis

All the following experiments were done using the [Inference Results](https://huggingface.co/datasets/bwarner/inference-scratch) from the [MedARC Benchmarking Suite](https://github.com/MedARC-AI/med-lm-envs)

## 1. Dynamics: Thinking Length & Failure
**Script:** `scripts/thinking_length.py`

Investigates if models "overthink" (generate excessive tokens) when they fail questions.

![Thinking Length](plots/thinking_length_correlation.png)

## 2. Robustness: Distractor Stress Test
**Script:** `scripts/distractor_test_2.py`

Measures performance degradation when an extra distractor option is added (4 vs. 5 options).

![Distractor Stress](plots/distractor_stress_test.png)

## 3. Potential: Pass@k Analysis
**Script:** `scripts/pass_at_k.py`

Compares baseline single-shot accuracy (Pass@1) against theoretical maximum potential (Pass@k) given multiple attempts.

![Pass@k](plots/pass_at_k_robust.png)

## 4. Audit: Signal vs. Noise
**Script:** `scripts/SNR.py`

Audits datasets for stability. Identifies "Lottery Zones" (high variance, low skill) versus reliable benchmarks.

![Signal to Noise](plots/signal_to_noise_audit.png)

## 5. Audit: Positional Bias
**Script:** `scripts/option_a_bias.py`

Checks for statistical preference for specific option letters (A-E) regardless of the correct answer.

![Positional Bias](plots/positional_bias_check.png)
