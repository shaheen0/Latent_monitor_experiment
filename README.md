# Small-model linear probes

I am exploring how simple detectors read a language model's internal vectors,
inspired by [*LLMs Learn to Evade Latent Monitors from Prior Feedback
Alone*](2609.36490v1.pdf).
I use small models because my older NVIDIA GPU has
4 GB of VRAM. My experiments cover **logistic regression only**, not the
paper's MLP or attention probes. I have not trained a LoRA adapter.

## What is included

- Response-vector extraction and logistic-regression probe training.
- Detection checks and paired label-imitation experiments with fixed response text.
- Saved results and reporting code.

I used the text examples shared in the paper authors' GitHub repository.
Each text group has 150 probe-training examples and 350 examples in `nf`
for evaluation and prompt demonstrations.
Main runs use 50 sentences per test group, with 16 target and 16 off-topic
examples in each prompt. Detection cutoffs stay fixed during comparisons.

## Results

Averages below cover ten test groups. AUROC uses 100 sentences per group;
detection rates and cosine use the 50 sentences in each main experiment.

| Model | Clean AUROC | Detected before editing | Detected at 8× edit | Detected at 32× edit | Mean cosine |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | 0.99929 | 98.4% | 97.6% | 90.2% | 0.00396 |
| SmolLM2-360M | 0.99962 | 99.0% | 97.8% | 77.0% | 0.04083 |
| Qwen2.5-0.5B | 0.99958 | 99.2% | 96.0% | 74.2% | 0.03108 |

The edits above are calculated as `h_clean − α(h_elevate − h_suppress)`.
Cosine measures how closely that paired difference points along the detector's
direction. These are offline vector edits, separate from the actual suppress
instruction results in the [full report](outputs/probe_report/aligned_evaluation/results.md).

The averages show strong clean detection and small positive cosine alignment,
but they hide differences between individual text groups. **For sarcasm in
actual episodes, Qwen2.5-0.5B's detection rate drops from 98% to 10%
(TPR 0.10) under the suppress instruction.** SmolLM2-135M still detects 86%
and SmolLM2-360M detects 82% under that instruction. Detection therefore varies
with the model, text group, and context in these runs. Qwen also drops to 8%
under the elevate instruction, so this is a monitor failure at the fixed
cutoff, not yet proof of deliberate or label-directed hiding.

I saved probes for 60 text groups with 135M and 47 with each of the other
models, so this is not a controlled comparison of model size. These results
do not establish that smaller models are safer.

![Results overview](outputs/probe_report/aligned_evaluation/overview.png)

Dashed lines are reference values from the paper for different models.
Shading shows 95% bootstrap intervals across the ten test groups.

## Tools I use

I use **PyTorch and Hugging Face Transformers** to load the models and
tokenizers and extract their internal vectors. I use **scikit-learn** for
logistic regression and metrics, **NumPy** for saved vectors, and
**Matplotlib** for figures.

The ML dependency versions are listed in [pyproject.toml](pyproject.toml).
