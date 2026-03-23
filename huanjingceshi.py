'EOF'
import torch
ckpt = torch.load("models/base.ckpt", map_location="cpu")
print(type(ckpt))
print(list(ckpt.keys())[:5])  # 看看里面有些什么 key
