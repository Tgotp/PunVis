# 配置文件
# 复制为 config.py 并填入你的API密钥

# OpenAI API配置
OPENAI_API_KEY = ""  # 填入你的OpenAI API密钥
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Anthropic Claude API配置
ANTHROPIC_API_KEY = ""

# 其他API配置（可选）
# 阿里云、百度等国内API

# 默认使用的模型
TEXT_MODEL = "gpt-4o"  # 或 "claude-3-5-sonnet-20241022"
VISION_MODEL = "gpt-4o"  # 用于看图猜成语的VLM

# 迭代参数
MAX_ITERATIONS = 5  # 最大迭代次数
MIN_CONFIDENCE = 0.8  # 最低信心度
