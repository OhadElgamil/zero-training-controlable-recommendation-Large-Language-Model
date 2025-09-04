import torch, json

payload = torch.load("./output/refusal_vec_item69.pt", map_location="cpu")
r = payload["r_hat"].float()       # the direction (unit-norm)
mu_pos = payload["mu_pos"].float()
mu_neg = payload["mu_neg"].float()
meta = payload["meta"]

print("shape:", tuple(r.shape))     # e.g., (4096,) for LLaMA-2-7B
print("‖r‖₂:", r.norm().item())
print("mean/std/min/max:", r.mean().item(), r.std().item(), r.min().item(), r.max().item())
print("first 16 dims:", r[:16])
print("meta:", json.dumps(meta, indent=2))