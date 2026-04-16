#!/usr/bin/env python
import nltk

# 下载停用词数据
print("Downloading NLTK stopwords...")
nltk.download('stopwords')

# 测试是否成功下载
from nltk.corpus import stopwords
print("English stopwords:")
print(stopwords.words('english')[:20])  # 打印前20个停用词
print(f"Total stopwords: {len(stopwords.words('english'))}")
print("Stopwords downloaded successfully!")