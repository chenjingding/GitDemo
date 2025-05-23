#"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载模型和分词器（假设已微调并保存为'sentiment_model'）
model_path = "C:\\Users\\15200\\PycharmProjects\\NLP\\bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 待分类句子
text = "剧情拖沓冗长，中途几次差点睡着。"

# 预处理和预测
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

# 输出结果（假设0=负面，1=正面）
label = "正面" if predictions.item() == 1 else "负面"
print(f"分类结果: {label}")
#"""
"""
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

text = "分量太少了，照片看着满满的，实际就几口。"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)  # 转换为概率
    print("负面概率:", probabilities[0][0].item(), "正面概率:", probabilities[0][1].item())
    # 根据业务逻辑手动调整阈值
    label = "正面" if probabilities[0][1] > 0.7 else "负面"  # 提高正面判断阈值
    print(f"分类结果: {label}")
"""
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载模型和分词器（假设已微调并保存为'sentiment_model'）
model_path = "C:\\Users\\15200\\PycharmProjects\\NLP\\bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 待分类句子
text = "分量太少了，照片看着满满的，实际就几口。"

# 预处理和预测
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

# 输出结果（假设0=负面，1=正面）
label = "正面" if predictions.item() == 1 else "负面"
print(f"分类结果: {label}")
"""