"""
核心Agent类
具备反思和学习能力的成语双关语生成Agent
"""

import json
import os
import base64
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import requests

from memory import ExperienceMemory, Experience


def check_pun_valid(idiom: str, pun: str) -> Tuple[bool, str]:
    """
    检查双关语是否有效
    要求：至少替换1个字，且长度相同
    注：同音检查由LLM保证，这里只做基础检查
    """
    # 清理双关语，去掉括号注释（如"叶公好笼（龙→笼）"）
    pun_clean = pun.split('（')[0].strip()

    if len(idiom) != len(pun_clean):
        return False, f"长度不一致: {idiom}({len(idiom)}) vs {pun_clean}({len(pun_clean)})"

    # 统计替换字数
    diff_count = sum(1 for c1, c2 in zip(idiom, pun_clean) if c1 != c2)

    if diff_count == 0:
        return False, "没有替换任何字（完全相同）"

    if diff_count < 1:
        return False, f"替换字数不足: 只替换了{diff_count}个"

    return True, f"有效双关: 替换了{diff_count}个字"


@dataclass
class GenerationAttempt:
    """生成尝试记录"""
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


class PunVisAgent:
    """成语双关语可视化Agent"""

    def __init__(self, client, memory: ExperienceMemory, skip_image: bool = False):
        """
        初始化Agent
        Args:
            client: LLM API客户端
            memory: 经验记忆系统
            skip_image: 是否跳过图像生成（如果API不支持DALL-E）
        """
        self.client = client
        self.memory = memory
        self.skip_image = skip_image

    def generate_with_reflection(
        self,
        idiom: str,
        max_iterations: int = 5,
        confidence_threshold: float = 0.8
    ) -> Tuple[bool, List[GenerationAttempt], str]:
        """
        带反思的生成流程

        Returns:
            (success, attempts, summary)
        """
        print(f"\n{'='*60}")
        print(f"🎯 Agent开始处理: {idiom}")
        print(f"{'='*60}")

        attempts = []
        history = []  # 记录历史尝试用于反思

        for iteration in range(1, max_iterations + 1):
            print(f"\n--- 第 {iteration} 轮迭代 ---")

            # 1. 读取记忆，获取经验教训
            memory_context = self._build_memory_context(idiom, history)

            # 2. 生成双关语和场景描述（应用经验）
            if iteration == 1:
                result = self._generate_initial(idiom, memory_context)
            else:
                # 获取上一轮反思的建议
                if attempts:
                    prev_attempt = attempts[-1]
                    prev_reflection = prev_attempt.get('reflection')
                    result = self._regenerate_with_lessons(
                        idiom, prev_attempt, prev_reflection, memory_context
                    )
                else:
                    result = None

            if not result:
                print("❌ 生成失败")
                continue

            pun, scene_zh, scene_en, reasoning = result

            # 🔍 检查双关语是否满足至少替换一个同音字
            is_valid, check_msg = check_pun_valid(idiom, pun)
            if not is_valid:
                print(f"❌ 双关语检查失败: {check_msg}")
                print(f"   原成语: {idiom}")
                print(f"   双关语: {pun}")
                print(f"   正在重新生成...")
                # 将错误信息加入历史，让LLM知道原因
                history.append({
                    'iteration': iteration,
                    'pun': pun,
                    'scene_zh': scene_zh,
                    'error': f"双关语无效: {check_msg}",
                    'is_correct': False
                })
                continue
            else:
                print(f"✅ {check_msg}")

            print(f"✅ 生成双关语: {pun}")
            print(f"📝 场景: {scene_zh[:80]}...")
            print(f"💭 思考: {reasoning[:100]}...")

            # 3. 生成图片
            print("🖼️ 生成图片...")
            image_path = self._generate_image(scene_en, idiom, iteration)
            if not image_path:
                print("❌ 图片生成失败，跳过本轮")
                continue
            print(f"✅ 图片已保存: {image_path}")

            # 4. 第一轮：图片质量评估
            print("🔍 第一轮：图片质量评估...")
            quality_result = self._evaluate_image_quality(image_path, scene_zh, pun, idiom)

            print(f"   总体得分: {quality_result['overall_score']:.0%}")
            print(f"   主体呈现: {quality_result['subject_score']:.0%}")
            print(f"   动作表达: {quality_result['action_score']:.0%}")
            print(f"   场景匹配: {quality_result['scene_score']:.0%}")
            print(f"   干扰元素: {quality_result['interference_score']:.0%}")
            print(f"   分析: {quality_result['analysis'][:100]}...")

            if not quality_result['passed']:
                print(f"   ❌ 图片质量未通过，跳过猜成语环节")
                # 构造一个失败的评估结果
                eval_result = {
                    'guessed': '图片质量不合格',
                    'confidence': 0.0,
                    'reasoning': f"图片质量检查未通过: {quality_result['analysis']}",
                    'is_correct': False,
                    'quality_passed': False
                }
            else:
                print(f"   ✅ 图片质量通过，进入第二轮...")
                # 5. 第二轮：猜成语
                print("🔍 第二轮：VLM猜成语...")
                eval_result = self._vlm_evaluate(image_path, idiom, pun)
                eval_result['quality_passed'] = True

                print(f"   猜测: {eval_result['guessed']}")
                print(f"   信心度: {eval_result['confidence']:.0%}")
                print(f"   推理: {eval_result['reasoning'][:120]}...")

            # 6. 反思分析
            attempt = {
                'iteration': iteration,
                'pun': pun,
                'scene_zh': scene_zh,
                'scene_en': scene_en,
                'image_path': image_path,
                'quality_result': quality_result,
                'vlm_guess': eval_result['guessed'],
                'vlm_confidence': eval_result['confidence'],
                'vlm_reasoning': eval_result['reasoning'],
                'is_correct': eval_result['is_correct'],
                'quality_passed': eval_result.get('quality_passed', False)
            }

            reflection = self._reflect(idiom, attempt, history, memory_context)
            attempt['reflection'] = reflection
            attempts.append(attempt)
            history.append(attempt)

            print(f"🤔 反思: {reflection.lesson_learned[:100]}...")

            # 6. 如果成功，保存经验并结束
            if eval_result['is_correct'] and eval_result['confidence'] >= confidence_threshold:
                print(f"\n🎉 成功！VLM正确猜出成语")
                # 传入history[:-1]因为当前attempt还没被加入history（虽然实际上已经加入了，但为了正确判断进步，需要传入不包含当前attempt的history）
                # 实际上attempt已经在上面被加入到history了，所以我们需要调整逻辑
                # 创建不包含当前attempt的历史副本用于判断
                history_for_comparison = history[:-1] if history else []
                self._save_experience(idiom, attempt, reflection, history_for_comparison)
                return True, attempts, self._generate_summary(attempts, True)

            # 7. 如果失败但还有迭代次数，继续优化
            print(f"⚠️ 未猜中，继续优化...")

        # 所有迭代都失败
        print(f"\n❌ 达到最大迭代次数，未能成功")
        if attempts:
            self._save_experience(idiom, attempts[-1], attempts[-1]['reflection'], history)
        return False, attempts, self._generate_summary(attempts, False)

    def _build_memory_context(self, idiom: str, history: List[Dict]) -> str:
        """构建记忆上下文"""
        context_parts = []

        # 添加已总结的规则
        rules_text = self.memory.get_rules_text()
        context_parts.append(rules_text)

        # 添加成功经验
        success_patterns = self.memory.get_success_patterns()
        context_parts.append(success_patterns)

        # 添加示例参考
        examples_text = self._load_examples_text()
        if examples_text:
            context_parts.append(examples_text)

    def _load_examples_text(self) -> str:
        """加载示例文件作为参考"""
        try:
            with open("examples/chengyu_examples.md", "r", encoding="utf-8") as f:
                content = f.read()

            # 提取表格内容
            lines = content.split('\n')
            examples = []
            for line in lines:
                # 匹配表格行 | 原词 | 双关词 | 场景描述 | 英文 |
                if line.startswith('|') and '原词' not in line and '---' not in line:
                    parts = [p.strip() for p in line.split('|')]
                    parts = [p for p in parts if p]
                    if len(parts) >= 3:
                        examples.append({
                            'original': parts[0],
                            'pun': parts[1],
                            'scene': parts[2]
                        })

            if examples:
                lines = ["\n## 参考示例\n"]
                for e in examples[:5]:  # 只显示前5个
                    lines.append(f"- {e['original']} → {e['pun']}: {e['scene'][:50]}...")
                return "\n".join(lines)
        except:
            pass
        return ""
        if history:
            context_parts.append("\n## 本次尝试历史\n")
            for h in history:
                context_parts.append(f"- 第{h['iteration']}轮: 猜的是'{h['vlm_guess']}' {'✅' if h['is_correct'] else '❌'}")

        return "\n".join(context_parts)

    def _generate_initial(self, idiom: str, memory_context: str) -> Optional[Tuple]:
        """初始生成"""
        system_prompt = f"""你是一位精通中文双关语的创意专家。请将成语转换为同音双关语，并生成场景描述。

{memory_context}

## 🚨 最高优先级约束（必须严格遵守）
**绝对禁止场景中出现任何文字、标牌、字母、符号、书法、刻字！**
- ❌ 禁止：纸张上的字、牌匾、指示牌、标签、印章、涂鸦
- ❌ 禁止：用物体摆放成文字形状
- ❌ 禁止：天空/云层中形成文字
- ✅ 允许：纯视觉元素（动作、物体、场景）

## 双关语要求（必须满足！）
1. **至少替换1个字**：必须将原成语中的至少一个字替换为同音字
2. **保持同音**：替换后的字必须与原字读音相同（声调可以不同）
3. **示例**：机不可失 → 鸡不可湿（机→鸡），对牛弹琴 → 对牛谈情（琴→情）

## 场景要求
- 只用纯视觉元素（动作、物体、表情）表达双关含义
- 通过动作和物体关系表达含义，而非文字

## 参考示例风格
- 鸡不可湿 → 小鸡撑伞避雨（动作表达）
- 守猪待兔 → 农夫抱猪等兔子
- 对牛谈情 → 人向牛表白，牛害羞

输出JSON格式：
{{
    "pun": "双关语（必须至少替换1个同音字）",
    "scene_zh": "中文场景描述",
    "scene_en": "英文场景描述",
    "reasoning": "生成思路说明（说明替换了哪些字，为什么选择这些字）"
}}"""

        user_prompt = f"成语：{idiom}\n\n请生成双关语和场景描述。"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return (
                result.get("pun", ""),
                result.get("scene_zh", ""),
                result.get("scene_en", ""),
                result.get("reasoning", "")
            )
        except Exception as e:
            print(f"生成错误: {e}")
            return None

    def _regenerate_with_lessons(
        self,
        idiom: str,
        prev_attempt: Dict,
        reflection: ReflectionResult,
        memory_context: str
    ) -> Optional[Tuple]:
        """基于反思重新生成"""
        system_prompt = f"""你是一位精通中文双关语的创意专家。上一轮生成失败了，请根据反思和已总结的规则改进。

{memory_context}

## 🚨 最高优先级约束（必须严格遵守）
**绝对禁止场景中出现任何文字、标牌、字母、符号、书法、刻字！**
- ❌ 禁止：纸张上的字、牌匾、指示牌、标签、印章、涂鸦
- ❌ 禁止：用物体摆放成文字形状
- ❌ 禁止：天空/云层中形成文字
- ✅ 允许：纯视觉元素（动作、物体、场景）
- 只用动作和物体关系表达含义，绝不使用文字

## 上一轮的问题
- 双关语: {prev_attempt['pun']}
- 场景: {prev_attempt['scene_zh']}
- VLM猜成了: {prev_attempt['vlm_guess']}
- VLM的思考: {prev_attempt['vlm_reasoning']}

## 反思总结
- 问题分析: {reflection.analysis}
- 关键问题: {', '.join(reflection.key_factors)}
- 改进建议: {', '.join(reflection.suggestions)}

请根据以上信息，生成改进后的双关语和场景。

输出JSON格式：
{{
    "pun": "新的双关语",
    "scene_zh": "改进后的中文场景描述（禁止任何文字、标牌、字母）",
    "scene_en": "改进后的英文场景描述",
    "reasoning": "改进思路说明",
    "changes": "具体改了什么"
}}"""

        user_prompt = f"成语：{idiom}\n\n请根据反思改进生成。"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.9,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return (
                result.get("pun", prev_attempt['pun']),
                result.get("scene_zh", ""),
                result.get("scene_en", ""),
                result.get("reasoning", "") + f" [改进: {result.get('changes', '')}]"
            )
        except Exception as e:
            print(f"重新生成错误: {e}")
            return None

    def _reflect(
        self,
        idiom: str,
        current: Dict,
        history: List[Dict],
        memory_context: str
    ) -> ReflectionResult:
        """反思分析"""
        is_correct = current['is_correct']
        quality_passed = current.get('quality_passed', True)
        quality_result = current.get('quality_result', {})

        # 根据是否通过质量检查，选择不同的反思重点
        if not quality_passed:
            # 质量未通过，重点分析图片质量问题
            system_prompt = f"""你是一位图片质量分析专家。图片未能通过质量检查，请分析问题所在。

## 质量检查结果
- 总体得分: {quality_result.get('overall_score', 0):.0%}
- 主体呈现: {quality_result.get('subject_score', 0):.0%}
- 动作表达: {quality_result.get('action_score', 0):.0%}
- 场景匹配: {quality_result.get('scene_score', 0):.0%}
- 干扰元素: {quality_result.get('interference_score', 0):.0%}
- 质量分析: {quality_result.get('analysis', '无')}

## 场景描述
{current['scene_zh']}

## 任务
请分析为什么生成的图片不符合场景描述，导致质量检查失败。
重点关注：
1. 场景描述是否过于复杂，导致AI画图时遗漏关键元素
2. 是否有文字/标识等不应该出现的干扰元素
3. 主体和动作是否与描述一致
4. 场景描述是否需要简化和聚焦

输出JSON格式：
{{
    "analysis": "图片质量问题的具体分析",
    "key_factors": ["问题1", "问题2"],
    "suggestions": ["改进建议1", "改进建议2"],
    "lesson_learned": "一句话总结：下次如何改进场景描述"
}}"""
        else:
            # 质量通过但猜错，分析双关语和场景表达问题
            system_prompt = f"""你是一位成语双关语创作批评专家。请分析为什么VLM没能猜出正确的成语。

## 分析原则（只能从以下角度分析）
1. **双关词问题**：同音替换是否自然？是否容易从视觉上联想到原成语？
2. **场景描述问题**：
   - 主体是否明确突出？
   - 动作是否能清晰表达"双关"含义？
   - 是否有干扰元素导致误解？
   - 场景是否太复杂或太简单？
3. **视觉表达问题**：画面元素是否能引导观者联想到目标成语？

## 禁止分析的内容
- 不要分析API模型的问题
- 不要批评VLM的能力
- 只关注双关词和场景描述本身的问题

## 当前尝试
- 原成语: {idiom}
- 双关语: {current['pun']}
- 场景描述: {current['scene_zh']}
- VLM猜成了: {current['vlm_guess']}
- VLM的思考: {current['vlm_reasoning']}
- 是否正确: {'是' if is_correct else '否'}

## 任务
请深入分析为什么{'成功' if is_correct else '失败'}，**只能**从双关词和场景描述的角度分析原因，给出具体的改进建议。

输出JSON格式：
{{
    "analysis": "为什么没猜中？从双关词和场景描述角度分析",
    "key_factors": ["问题因素1", "问题因素2"],
    "suggestions": ["改进建议1", "改进建议2"],
    "lesson_learned": "一句话总结：下次应该怎么做"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "请进行反思分析。"}
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return ReflectionResult(
                success=is_correct,
                analysis=result.get("analysis", ""),
                key_factors=result.get("key_factors", []),
                suggestions=result.get("suggestions", []),
                lesson_learned=result.get("lesson_learned", "")
            )
        except Exception as e:
            print(f"反思错误: {e}")
            return ReflectionResult(
                success=is_correct,
                analysis="反思失败",
                key_factors=[],
                suggestions=["重新尝试"],
                lesson_learned="需要更多尝试"
            )

    def _save_experience(self, idiom: str, attempt: Dict, reflection: ReflectionResult, history: List[Dict]):
        """
        智能保存经验：只有这次比上一轮有进步时才保存

        进步标准：
        1. 猜对了（is_correct=True）
        2. 信心度比上一轮高
        3. 猜的成语更接近目标（如果上一轮完全猜错）
        """
        is_improvement = False
        improvement_reason = ""

        if len(history) == 0:
            # 第一轮，总是有意义的
            is_improvement = True
            improvement_reason = "第一轮尝试"
        else:
            prev_attempt = history[-1]

            # 检查是否有进步
            if attempt['is_correct'] and not prev_attempt['is_correct']:
                # 从错到对
                is_improvement = True
                improvement_reason = "成功猜对（上一轮错误）"
            elif attempt['is_correct'] and prev_attempt['is_correct']:
                # 都对，但信心度更高
                if attempt['vlm_confidence'] > prev_attempt['vlm_confidence']:
                    is_improvement = True
                    improvement_reason = f"信心度提升 {prev_attempt['vlm_confidence']:.0%} → {attempt['vlm_confidence']:.0%}"
            elif not attempt['is_correct'] and not prev_attempt['is_correct']:
                # 都错，检查质量是否有提升
                curr_quality = attempt.get('quality_result', {}).get('overall_score', 0)
                prev_quality = prev_attempt.get('quality_result', {}).get('overall_score', 0)

                if curr_quality > prev_quality:
                    is_improvement = True
                    improvement_reason = f"图片质量提升 {prev_quality:.0%} → {curr_quality:.0%}"
                elif idiom in attempt['vlm_guess'] and idiom not in prev_attempt['vlm_guess']:
                    is_improvement = True
                    improvement_reason = "猜测更接近目标"
                elif attempt['vlm_confidence'] < prev_attempt['vlm_confidence'] and curr_quality <= prev_quality:
                    # 信心度下降且质量没提升，不保存
                    is_improvement = False
                    improvement_reason = f"信心度下降且无质量提升，跳过保存"
                else:
                    # 都错，但质量持平或有其他进步
                    is_improvement = True
                    improvement_reason = f"第{attempt['iteration']}轮尝试（累积经验）"

        if not is_improvement:
            print(f"⏭️ 跳过经验保存: {improvement_reason}")
            return

        # 判断成功/失败
        success = attempt['is_correct']

        # 构建包含质量信息的 reason
        quality_info = ""
        if 'quality_result' in attempt:
            q = attempt['quality_result']
            quality_info = f"[质量: {q.get('overall_score', 0):.0%}, 通过: {q.get('passed', False)}] "

        exp = Experience(
            idiom=idiom,
            pun=attempt['pun'],
            scene_zh=attempt['scene_zh'],
            success=success,
            iteration=attempt['iteration'],
            reason=f"{quality_info}{reflection.analysis} [{improvement_reason}]",
            key_factors=reflection.key_factors,
            timestamp=datetime.now().isoformat(),
            vlm_feedback=attempt['vlm_reasoning']
        )
        self.memory.add_experience(exp)

        # 只有真正有新洞察时才提取规则
        if success or self._is_new_insight(reflection, history):
            self._extract_rules_from_reflection(reflection, success)

    def _is_new_insight(self, reflection: ReflectionResult, history: List[Dict]) -> bool:
        """
        判断这次反思是否提供了新的洞察（而非重复已知问题）
        """
        if len(history) == 0:
            return True

        # 获取之前所有反思的关键因素
        prev_factors = set()
        for h in history:
            if 'reflection' in h:
                prev_factors.update(h['reflection'].key_factors)

        # 检查是否有新因素
        current_factors = set(reflection.key_factors)
        new_factors = current_factors - prev_factors

        if new_factors:
            print(f"💡 发现新问题: {new_factors}")
            return True
        else:
            print(f"⏭️ 问题与之前重复: {current_factors}")
            return False

    def _extract_rules_from_reflection(self, reflection: ReflectionResult, success: bool):
        """调用LLM从反思中总结通用规则，避免重复"""
        if not self.client:
            return

        # 获取已有规则名称，避免重复
        existing_rules = set(self.memory.rules.keys())

        system_prompt = f"""你是一位规则总结专家。请从本次成语双关语生成的反思中，总结出可以指导未来生成的通用规则。

当前已有规则：{', '.join(existing_rules)}

总结要求：
1. 规则要通用，不仅适用于本次成语
2. 规则要具体、可操作
3. **重要**：如果反思中的要点已经被已有规则覆盖，不要重复添加
4. 只添加真正新的、有价值的规则（最多1-2条）
5. 如果没有新规则需要添加，返回空对象 {{}}

规则格式：{{"规则名称": "规则具体内容"}}

示例：
- 反思提到"主体不够突出"且没有"主体明确"规则 → {{"主体明确": "场景中必须有清晰、占据主要画面的主体"}}
- 反思提到"出现了干扰元素"且没有相关规则 → {{"避免干扰": "移除可能引起其他成语联想的元素"}}
- 如果已有"场景简洁"规则，反思也提到简洁问题 → 返回 {{}}

只输出JSON格式，不要其他内容：
{{"规则1": "内容"}} 或 {{}}"""

        user_prompt = f"""本次生成：{'成功' if success else '失败'}
反思分析：{reflection.analysis}
关键因素：{', '.join(reflection.key_factors)}
改进建议：{', '.join(reflection.suggestions)}

请总结通用规则（如果没有新规则，返回空对象）。"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )

            new_rules = json.loads(response.choices[0].message.content)

            # 过滤掉已存在的规则
            truly_new_rules = {}
            for name, desc in new_rules.items():
                if name not in existing_rules:
                    truly_new_rules[name] = desc

            if truly_new_rules:
                self.memory.update_rules(truly_new_rules)
                print(f"💡 LLM总结出 {len(truly_new_rules)} 条新规则: {list(truly_new_rules.keys())}")
            else:
                print(f"💡 没有新规则需要添加（已被现有规则覆盖）")

        except Exception as e:
            print(f"规则提取失败: {e}")

    def _generate_summary(self, attempts: List[Dict], success: bool) -> str:
        """生成总结"""
        total = len(attempts)
        if not attempts:
            return "未能完成任何迭代。"

        # 获取最后一轮的详细质量信息
        final = attempts[-1]
        quality = final.get('quality_result', {})
        quality_str = f"(图片质量: {quality.get('overall_score', 0):.0%})"

        if success:
            return f"成语'{final['pun']}'经过{total}轮迭代成功生成可识别的视觉双关语。{quality_str}"
        else:
            if not final.get('quality_passed', True):
                return f"成语经过{total}轮迭代，最后一轮图片质量检查未通过{quality_str}，建议优化场景描述。"
            return f"成语经过{total}轮迭代未能成功{quality_str}，建议更换双关思路或人工优化。"

    def _generate_image(self, scene_en: str, idiom: str, iteration: int) -> str:
        """生成图片 - 使用z-image-turbo"""
        try:
            style = "Bright cartoon style, clean background, simple composition, cute and expressive"
            prompt = f"{scene_en}. {style}"

            print(f"   调用z-image-turbo生成图片...")
            print(f"   提示词: {prompt[:80]}...")

            response = self.client.images.generate(
                model="z-image-turbo",
                prompt=prompt,
                size="1024x1024",
                n=1,
            )

            image_url = response.data[0].url
            if not image_url:
                print(f"   ❌ API返回空URL")
                return ""

            print(f"   图片URL获取成功")

            # 下载图片
            image_data = requests.get(image_url, timeout=60).content

            # 保存
            os.makedirs("output/images", exist_ok=True)
            image_path = f"output/images/{idiom}_v{iteration}.png"
            with open(image_path, "wb") as f:
                f.write(image_data)

            print(f"   ✅ 图片已保存: {image_path}")
            return image_path

        except Exception as e:
            print(f"❌ 图片生成失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _evaluate_image_quality(self, image_path: str, scene_zh: str, pun: str, idiom: str) -> Dict:
        """第一轮：评估图片质量，检查是否符合场景描述"""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            system_prompt = f"""你是一位图片质量评估专家。请严格评估生成的图片是否符合预期的场景描述。

## 🚨 最高优先级检查（一票否决）
**图片中绝对不能出现任何文字、字母、标牌、符号、书法、刻字！**
- 检查是否有纸张上的文字、牌匾、指示牌、标签
- 检查是否有云朵/物体排列成文字形状
- 检查是否有任何可识别的字母或汉字
- 如果发现任何文字，干扰元素得分为0，直接判定为不通过

## 其他评估维度
1. **主体呈现** (0-1分): 图片中的主要主体是否与描述一致
2. **动作表达** (0-1分): 主体的动作/行为是否与描述一致
3. **场景匹配** (0-1分): 整体场景氛围是否与描述匹配
4. **干扰元素** (0-1分): 是否有不应出现的文字、多余元素等（无文字/干扰=1分）

## 场景描述
{scene_zh}

## 预期双关语
{pun} (原成语: {idiom})

## 输出格式
{{
    "overall_score": "总体得分(0-1，四项平均)",
    "subject_score": "主体呈现得分",
    "action_score": "动作表达得分",
    "scene_score": "场景匹配得分",
    "interference_score": "干扰元素得分(有文字=0, 无干扰=1)",
    "text_found": "是否发现任何文字(true/false)",
    "passed": "是否通过(总体得分>=0.7且干扰元素>=0.8且无文字)",
    "analysis": "详细分析: 图片中实际看到了什么，与描述的差异，特别说明是否发现文字"
}}"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请评估这张图片是否符合场景描述。请以json格式输出。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # 计算是否通过
            overall = float(result.get("overall_score", 0))
            interference = float(result.get("interference_score", 0))
            text_found = result.get("text_found", False)

            # 如果发现文字，直接判定不通过
            if text_found:
                passed = False
                print(f"   🚫 检测到图片中出现文字，判定不通过！")
            else:
                passed = overall >= 0.7 and interference >= 0.8

            return {
                "overall_score": overall,
                "subject_score": float(result.get("subject_score", 0)),
                "action_score": float(result.get("action_score", 0)),
                "scene_score": float(result.get("scene_score", 0)),
                "interference_score": interference,
                "text_found": text_found,
                "passed": passed,
                "analysis": result.get("analysis", "")
            }

        except Exception as e:
            print(f"❌ 图片质量评估失败: {e}")
            return {"passed": False, "overall_score": 0, "analysis": f"评估失败: {e}"}

    def _vlm_evaluate(self, image_path: str, idiom: str, pun: str) -> Dict:
        """第二轮：VLM看图猜成语"""
        try:
            # 读取图片
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            system_prompt = f"""你是一位猜成语专家。请根据看到的图片，猜测这是对应哪个成语。

## 重要规则
这张图是用**同音双关**的方式表达的成语！
- 图片中的文字/场景是**谐音替换**后的版本
- 你需要从画面的视觉元素联想到**原成语**
- 例如：看到"鸡"可能要想到"机"，看到"湿"可能要想到"失"

## 猜测方法
1. 观察画面中的主体（动物、人物、物体）
2. 思考这些主体可能谐音替换了原成语中的哪些字
3. 结合动作和场景，联想原成语的含义
4. 输出你认为最可能的**原成语**（不是画面中的谐音版本）

## 输出格式
{{
    "guessed": "原成语（不是谐音版本）",
    "confidence": "信心度（0-1之间的数字）",
    "reasoning": "思考过程：看到了什么画面元素，如何谐音联想到原成语"
}}"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "请分析这张图片，猜测对应的成语。请以json格式输出。"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            guessed = result.get("guessed", "").strip()
            confidence = float(result.get("confidence", 0))
            reasoning = result.get("reasoning", "")

            # 判断是否猜对（包含匹配）
            is_correct = idiom in guessed or guessed in idiom

            return {
                "guessed": guessed,
                "confidence": confidence,
                "reasoning": reasoning,
                "is_correct": is_correct
            }

        except Exception as e:
            print(f"❌ VLM评估失败: {e}")
            return {
                "guessed": "",
                "confidence": 0.0,
                "reasoning": f"评估出错: {str(e)}",
                "is_correct": False
            }
