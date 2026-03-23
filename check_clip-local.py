from transformers import CLIPTokenizer, CLIPTextModel
import os

# 修改成你的本地路径，推荐绝对路径
clip_path = r"D:/Pycharm24/cdad/One-for-More/models/clip-vit-large-patch14"

print(f"检查路径是否存在: {clip_path}")
print("包含文件:", os.listdir(clip_path))

try:
    # 加载 tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(clip_path)
    print("✅ 成功加载 tokenizer")

    # 加载 text model
    text_model = CLIPTextModel.from_pretrained(clip_path)
    print("✅ 成功加载 CLIPTextModel")

    # 简单测试
    inputs = tokenizer("Hello anomaly detection!", return_tensors="pt")
    outputs = text_model(**inputs)
    print("✅ 成功运行 CLIPTextModel，输出 shape:", outputs.last_hidden_state.shape)

except Exception as e:
    print("❌ 出错啦:", e)
