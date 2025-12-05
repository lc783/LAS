import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import CLIPProcessor, CLIPModel

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
total_params  = clip.num_parameters()
# 设置模型路径
model_path = '/data/llm/longchen/Yi-6B-Chat'

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # 自动选择可用的设备
    torch_dtype='auto'  # 自动选择合适的数据类型
).eval()  # 将模型设置为评估模式

print(model)
print("Model loading complete!")