python -m tools.apply_weight_ablation \
  --llm_path  meta-llama/Llama-2-7b-hf \
  --vec_path ./output/refusal_vec_item69.pt \
  --beta 1.0 \
  --save_dir ./llama_patched_item69_beta1
