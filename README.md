# 成语双关语可视化Agent

一个具备**经验学习**能力的Agent系统，能将成语转化为易于理解的视觉双关语。

## 核心特性

### 🤖 Agent能力
- **经验记忆**: 自动保存每次尝试的成功/失败经验
- **反思学习**: 每次迭代后分析原因，提取可复用经验
- **经验应用**: 生成新内容时自动参考历史经验
- **持续积累**: 经验可以跨会话积累，越用越聪明

### 🧠 工作流程

```
输入成语
    ↓
Agent读取历史经验
    ↓
LLM生成双关语 + 场景（应用经验）
    ↓
生成图片
    ↓
VLM看图猜成语
    ↓
Agent反思分析（成功/失败原因）
    ↓
保存经验到Memory
    ↓
  ├─ 猜对了 → ✅ 记录成功经验，结束
  └─ 猜错了 → 💡 应用经验教训，重新生成
              ↓
        重复直到成功或达最大迭代次数
```

## 项目结构

```
punvis/
├── src/
│   ├── config.py          # API配置
│   ├── memory.py          # 经验记忆系统 ⭐
│   ├── agent.py           # 核心Agent（反思+学习）⭐
│   └── main.py            # 入口
├── memory/                # 经验存储目录 ⭐
│   ├── experiences.json   # 经验记录
│   └── rules.json         # 总结的规则
├── examples/
│   ├── PunBenchmark.json  # 示例数据集
│   └── chengyu_examples.md
└── output/
    └── images/            # 生成的图片
```

## 快速开始

### 1. 配置API

```bash
cp src/config_template.py src/config.py
# 编辑 src/config.py 填入API密钥
```

### 2. 运行

```bash
cd /Users/tgotp/program/punvis
python src/main.py --idiom "机不可失"
```

### 3. 查看经验

运行后会在 `memory/experiences.json` 中保存经验记录。

## Memory系统

### 经验记录格式

```json
{
  "idiom": "机不可失",
  "pun": "鸡不可湿",
  "scene_zh": "一只小鸡在下雨天撑着伞...",
  "success": true,
  "iteration": 2,
  "reason": "主体明确（小鸡撑伞），动作清晰（避雨），同音自然",
  "key_factors": ["主体明确", "动作清晰", "同音自然"],
  "timestamp": "2024-01-15T10:30:00",
  "vlm_feedback": "看到了小鸡和伞，联想到鸡..."
}
```

### Agent如何使用经验

1. **生成前**: 读取相关成语的经验，避免重复错误
2. **生成时**: LLM prompt中包含历史经验和成功模式
3. **反思后**: 提取关键教训，更新经验库
4. **持续优化**: 随着经验积累，生成质量不断提升

## 示例对话

**第一轮尝试:**
```
Agent: 生成双关语 "鸡不可湿"
VLM: 猜成 "鸡犬不宁" ❌
Agent反思: 问题在于场景没有体现"失/湿"的对比
保存经验: 需要增加"湿vs干"的视觉对比
```

**第二轮尝试:**
```
Agent: 读取经验，增加"小鸡身上干的，周围下雨"的对比
生成新场景: "小鸡在雨中紧紧抱着伞，身上有阳光..."
VLM: 猜成 "机不可失" ✅
Agent反思: 成功！关键在于视觉对比明确
保存经验: 视觉对比是关键成功因素
```

## 配置说明

`src/config.py`:

```python
OPENAI_API_KEY = "your-key"
OPENAI_BASE_URL = "https://jusuan.ai/v1"  # 如果使用代理
```

## 依赖

```bash
pip install openai
```

## 未来扩展

- [ ] 经验可视化分析
- [ ] 批量处理成语集
- [ ] 集成真实图像生成API
- [ ] 经验导出为训练数据
