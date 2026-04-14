#!/bin/bash

# 成语双关语可视化生成器启动脚本

# 检查配置文件
if [ ! -f "src/config.py" ]; then
    echo "错误：请先创建 src/config.py 文件"
    echo "复制 src/config_template.py 为 src/config.py 并填入你的API密钥"
    exit 1
fi

# 安装依赖
pip install -r requirements.txt -q

# 运行程序
python src/main.py "$@"
