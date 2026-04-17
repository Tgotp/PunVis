# 成语双关语可视化 Agent

一个面向成语双关语可视化的闭环 Agent Demo。系统已打通从成语输入到图像生成、视觉评估、反思迭代和经验沉淀的完整流程。

## 项目状态

- 当前状态：已完成端到端可运行 Demo（结项）
- 主要目标：验证流程可运行性与闭环可迭代性
- 当前定位：工程原型，已具备后续研究扩展基础

## 方法流程（简版）

```text
输入成语
  -> 读取历史经验
  -> 生成双关语与场景
  -> 生成图片
  -> VLM 看图猜成语
  -> 反思分析（成功/失败）
  -> 写回经验库
  -> 猜中则结束；未猜中则继续迭代
```

## 代码结构

```text
punvis/
├── src/
│   ├── main.py                # 入口
│   ├── agent.py               # 流程编排
│   ├── generation_service.py  # 文本生成相关
│   ├── vision_service.py      # 图像生成与视觉评估
│   ├── reflection_service.py  # 反思与经验写回
│   ├── memory.py              # 经验记忆系统
│   ├── agent_types.py
│   ├── agent_utils.py
│   ├── prompts.py
│   ├── config_template.py
│   └── config.py              # 本地配置（自行创建）
├── memory/
│   ├── experiences.json       # 经验记录
│   └── rules.json             # 规则总结
├── examples/
│   ├── PunBenchmark.json
│   └── chengyu_examples.md
├── image/                     # 成功示例图
└── output/
    └── images/                # 运行时输出
```

## 快速运行

1) 安装依赖

```bash
pip install -r requirements.txt
```

2) 创建配置文件

```bash
cp src/config_template.py src/config.py
```

在 `src/config.py` 中填写 API 配置。

3) 运行示例

```bash
python src/main.py --idiom "画龙点睛"
```

## 成功示例

当前仓库中的成功案例图位于 `image/` 目录：

- `image/���˷θ�-���˷θ�.png`
- `image/�����㾦-�������.png`
- `image/������ - ������.png`

在支持图片渲染的平台中可直接展示：

![示例1](image/���˷θ�-���˷θ�.png)
![示例2](image/�����㾦-�������.png)
![示例3](image/������ - ������.png)

## 结论

- 已验证“生成 -> 评估 -> 反思 -> 记忆”的闭环流程可行。
- 系统可在部分成语上实现低轮次收敛并成功猜回。
- 项目已达到“可演示、可复现实验流程”的阶段性目标。

