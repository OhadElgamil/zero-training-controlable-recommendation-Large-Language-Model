# Zero-Training Controllable Recommendation with Large Language Models

## Abstract
Large language model-based recommendation systems (LLMRec) have shown strong performance by leveraging powerful language representations for preference prediction. However, such systems often lack fine-grained controllability over targeted item exposure in practical settings (e.g., promoting new items, correcting underexposure, or conducting controlled interventions). This repository investigates a lightweight approach for manipulating item recommendation scores by applying targeted embedding perturbations—without retraining the full model and without intending to degrade overall performance. :contentReference[oaicite:1]{index=1}

---

## Method (Overview)
The central premise is that certain latent directions in the embedding space encode semantic/behavioral axes relevant to exposure or recommendation likelihood. By identifying an appropriate direction vector and perturbing target item embeddings along it, we can continuously control recommendation propensity at inference time. :contentReference[oaicite:2]{index=2}

### Models and Dataset
Experiments follow standard preprocessing on a benchmark movie-ratings dataset and evaluate two representative LLM-based recommenders:
- **TALLRec** (parameter-efficient tuning framework for LLM recommendation) :contentReference[oaicite:3]{index=3}  
- **LLaRA** (adapter tuning + prompt learning for LLM recommendation) :contentReference[oaicite:4]{index=4}  

Item embeddings \(v_i \in \mathbb{R}^d\) are extracted from the model’s final encoder layer to form the steering space. :contentReference[oaicite:5]{index=5}

---

## Direction Identification
Let \(I^+\) denote a set of “highly recommended / target” items and \(I^-\) a set of “neutral / low-exposure” items. A steering direction is computed via mean-difference:

$$
d = \frac{1}{|I^+|}\sum_{i\in I^+} v_i - \frac{1}{|I^-|}\sum_{i\in I^-} v_i
$$


Then normalize:

$$
\tilde{d}=\frac{d}{\|d\|_2}
$$


Optionally, PCA-based denoising can be applied to isolate the behavioral dimension. :contentReference[oaicite:6]{index=6}

---

## Directional Steering (Inference-Time Control)
Given \(\tilde{d}\), we modify a target item embedding via:

$$
v'_i = v_i + \alpha \tilde{d}
$$

and recompute scores \(s'(u,i)=f_\theta(u,v'_i)\). Positive \(\alpha\) increases a target item’s presence in a higher-exposure region; negative \(\alpha\) suppresses it. :contentReference[oaicite:7]{index=7}

### Practical Considerations
- **No retraining**: model weights remain unchanged. :contentReference[oaicite:8]{index=8}  
- **Low overhead**: direction computed offline once per intervention. :contentReference[oaicite:9]{index=9}  
- **Broad compatibility**: applicable to embedding-based recommenders with accessible item vectors. :contentReference[oaicite:10]{index=10}  

---

## Experimental Results (from attached PDF)

### Run 1: Polarity-Based Vector Definition
Steering direction computed from:
1) prompts where item “A” is the correct answer  
2) prompts where item “A” is not the correct answer :contentReference[oaicite:11]{index=11}

| Model Variant       | Hit@1(A) | Valid Ratio |
|--------------------|---------:|------------:|
| Clean (unsteered)  | 0.0755   | 0.5579      |
| Steered (enhanced) | 0.0690   | 0.6105      |

:contentReference[oaicite:12]{index=12}

### Run 2: Token-Based Vector Definition
Steering direction computed from:
1) prompts ending with “Answer: A”  
2) prompts ending with “Answer:” (without a specified item) :contentReference[oaicite:13]{index=13}

| Model Variant       | Hit@1(A) | Valid Ratio |
|--------------------|---------:|------------:|
| Clean (unsteered)  | 0.0755   | 0.5579      |
| Steered (enhanced) | 0.0443   | 0.5185      |

:contentReference[oaicite:14]{index=14}

**Interpretation.** The experiments show that steering alters model behavior, but the computed direction may not align cleanly with “recommendability” and can reduce Hit@1 depending on how contrast sets are defined. :contentReference[oaicite:15]{index=15}

