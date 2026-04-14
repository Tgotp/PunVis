"""
Agent经验记忆系统
存储和管理生成经验
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class Experience:
    """单个经验条目"""
    idiom: str                    # 原成语
    pun: str                     # 双关语
    scene_zh: str                # 场景描述
    success: bool                # 是否成功
    iteration: int               # 第几轮成功/失败
    reason: str                  # 成功/失败原因分析
    key_factors: List[str]       # 关键因素（正面或负面）
    timestamp: str               # 时间戳
    vlm_feedback: str            # VLM的原始反馈


class ExperienceMemory:
    """经验记忆系统"""

    MEMORY_FILE = "memory/experiences.json"
    RULES_FILE = "memory/rules.json"

    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.memory_path = os.path.join(project_root, self.MEMORY_FILE)
        self.rules_path = os.path.join(project_root, self.RULES_FILE)
        self.experiences: List[Experience] = []
        self.rules: Dict[str, Any] = {}

        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        self._load()

    def _load(self):
        """加载已有经验"""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.experiences = [Experience(**e) for e in data]
            print(f"📚 已加载 {len(self.experiences)} 条经验")

        if os.path.exists(self.rules_path):
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)

    def save(self):
        """保存经验到文件"""
        data = [asdict(e) for e in self.experiences]
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        with open(self.rules_path, 'w', encoding='utf-8') as f:
            json.dump(self.rules, f, ensure_ascii=False, indent=2)

    def add_experience(self, exp: Experience):
        """添加新经验"""
        self.experiences.append(exp)
        self.save()
        print(f"💾 已保存经验: {exp.idiom} - {'✅成功' if exp.success else '❌失败'}")

    def get_successful_cases(self, limit: int = 10) -> List[Experience]:
        """获取成功案例"""
        successes = [e for e in self.experiences if e.success]
        return successes[-limit:]  # 返回最近的

    def get_failed_cases(self, limit: int = 10) -> List[Experience]:
        """获取失败案例"""
        failures = [e for e in self.experiences if not e.success]
        return failures[-limit:]

    def update_rules(self, new_rules: Dict[str, str]):
        """更新规则库

        Args:
            new_rules: 从反思中提取的规则，如 {"主体明确": "场景必须有清晰的主体"}
        """
        self.rules.update(new_rules)
        self.save()
        print(f"📋 已更新规则: {list(new_rules.keys())}")

    def get_rules_text(self) -> str:
        """获取格式化的规则文本用于prompt"""
        if not self.rules:
            return "暂无规则。"
        lines = ["## 生成原则\n"]
        for name, desc in self.rules.items():
            lines.append(f"- **{name}**: {desc}")
        return "\n".join(lines)

    def get_formatted_memory(self) -> str:
        """获取格式化的记忆文本，用于prompt"""
        if not self.experiences:
            return "暂无经验记录。"

        recent = self.experiences[-10:]  # 最近10条
        lines = ["## 历史经验记录（最近10条）\n"]

        for i, e in enumerate(recent, 1):
            status = "✅成功" if e.success else "❌失败"
            lines.append(f"\n### 经验{i}: {e.idiom} -> {e.pun} [{status}]")
            lines.append(f"**场景**: {e.scene_zh[:80]}...")
            lines.append(f"**关键教训**: {', '.join(e.key_factors)}")
            lines.append(f"**原因分析**: {e.reason[:150]}...")

        return "\n".join(lines)

    def get_success_patterns(self) -> str:
        """提取成功模式"""
        successes = self.get_successful_cases(20)
        if not successes:
            return "暂无成功模式。"

        # 统计常见成功因素
        all_factors = []
        for e in successes:
            all_factors.extend(e.key_factors)

        factor_counts = {}
        for f in all_factors:
            factor_counts[f] = factor_counts.get(f, 0) + 1

        top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        lines = ["## 成功模式总结\n"]
        for factor, count in top_factors:
            lines.append(f"- {factor} (出现{count}次)")

        return "\n".join(lines)
