from dataclasses import dataclass
from typing import List


@dataclass
class GenerationAttempt:
    """生成尝试记录（兼容原返回类型注解）"""
    iteration: int
    pun: str
    scene_zh: str
    scene_en: str
    image_path: str
    vlm_guess: str
    vlm_confidence: float
    vlm_reasoning: str
    is_correct: bool


@dataclass
class ReflectionResult:
    """反思结果"""
    success: bool
    analysis: str
    key_factors: List[str]
    suggestions: List[str]
    lesson_learned: str
