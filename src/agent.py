"""
核心Agent类（编排层）
将生成、视觉评估、反思逻辑拆分到独立服务模块
"""

from typing import List, Tuple

from agent_types import GenerationAttempt
from agent_utils import check_pun_valid
from generation_service import build_memory_context, generate_initial, regenerate_with_lessons
from memory import ExperienceMemory
from reflection_service import generate_summary, reflect_attempt, save_experience
from vision_service import (
    evaluate_image_quality,
    generate_image,
    vlm_diagnose_with_answer,
    vlm_evaluate,
    vlm_forced_choice,
)


class PunVisAgent:
    """成语双关语可视化Agent（流程编排）"""

    def __init__(
        self,
        client,
        memory: ExperienceMemory,
        skip_image: bool = False,
        text_model: str = "gpt-4o",
        vision_model: str = "gpt-4o",
    ):
        self.client = client
        self.memory = memory
        self.skip_image = skip_image
        self.text_model = text_model
        self.vision_model = vision_model

    def generate_with_reflection(
        self, idiom: str, max_iterations: int = 5, confidence_threshold: float = 0.8
    ) -> Tuple[bool, List[GenerationAttempt], str]:
        print(f"\n{'='*60}")
        print(f"🎯 Agent开始处理: {idiom}")
        print(f"{'='*60}")

        attempts = []
        history = []

        for iteration in range(1, max_iterations + 1):
            print(f"\n--- 第 {iteration} 轮迭代 ---")
            memory_context = build_memory_context(self.memory, history)

            if iteration == 1 or not attempts:
                result = generate_initial(self.client, idiom, memory_context, self.text_model)
            else:
                result = regenerate_with_lessons(
                    self.client, idiom, attempts[-1], attempts[-1]["reflection"], memory_context, self.text_model
                )

            if not result:
                print("❌ 生成失败")
                continue

            pun, scene_zh, scene_en, reasoning = result
            is_valid, check_msg = check_pun_valid(idiom, pun)
            if not is_valid:
                print(f"❌ 双关语检查失败: {check_msg}")
                print(f"   原成语: {idiom}")
                print(f"   双关语: {pun}")
                print("   正在重新生成...")
                history.append({"iteration": iteration, "pun": pun, "scene_zh": scene_zh, "error": f"双关语无效: {check_msg}", "is_correct": False})
                continue

            print(f"✅ {check_msg}")
            print(f"✅ 生成双关语: {pun}")
            print(f"📝 场景: {scene_zh[:80]}...")
            print(f"💭 思考: {reasoning[:100]}...")

            print("🖼️ 生成图片...")
            image_path = generate_image(self.client, scene_en, idiom, iteration)
            if not image_path:
                print("❌ 图片生成失败，跳过本轮")
                continue
            print(f"✅ 图片已保存: {image_path}")

            print("🔍 第一轮：图片质量评估...")
            quality_result = evaluate_image_quality(
                self.client, image_path, scene_zh, pun, idiom, self.vision_model
            )
            print(f"   总体得分: {quality_result['overall_score']:.0%}")
            print(f"   主体呈现: {quality_result['subject_score']:.0%}")
            print(f"   动作表达: {quality_result['action_score']:.0%}")
            print(f"   场景匹配: {quality_result['scene_score']:.0%}")
            print(f"   干扰元素: {quality_result['interference_score']:.0%}")
            print(f"   分析: {quality_result['analysis'][:100]}...")

            if not quality_result["passed"]:
                print("   ❌ 图片质量未通过，跳过猜成语环节")
                eval_result = {
                    "guessed": "图片质量不合格",
                    "confidence": 0.0,
                    "reasoning": f"图片质量检查未通过: {quality_result['analysis']}",
                    "is_correct": False,
                    "quality_passed": False,
                }
            else:
                print("   ✅ 图片质量通过，进入第二轮...")
                print("🔍 第二轮：VLM猜成语...")
                eval_result = vlm_evaluate(self.client, image_path, idiom, self.vision_model)
                eval_result["quality_passed"] = True
                print(f"   猜测: {eval_result['guessed']}")
                print(f"   信心度: {eval_result['confidence']:.0%}")
                print(f"   推理: {eval_result['reasoning'][:120]}...")

                print("🔍 第三轮：VLM带答案诊断...")
                diagnosis_result = vlm_diagnose_with_answer(
                    self.client, image_path, idiom, pun, scene_zh, self.vision_model
                )
                print(f"   主体明确性: {diagnosis_result['subject_clarity']:.0%}")
                print(f"   视觉可读性: {diagnosis_result['visual_readability']:.0%}")
                print(f"   双关映射度: {diagnosis_result['pun_mapping']:.0%}")
                print(f"   误导风险: {diagnosis_result['misleading_risk']:.0%}")
                print(f"   诊断结论: {diagnosis_result['analysis'][:120]}...")
                eval_result["diagnosis"] = diagnosis_result

                # 第四轮：候选强制选择（目标词 + 历史失败词）
                past_failed = []
                for h in history:
                    guess = h.get("vlm_guess", "")
                    if guess and guess != idiom and guess != "图片质量不合格":
                        past_failed.append(guess)
                candidates = list(dict.fromkeys([idiom, eval_result["guessed"], *past_failed]))[:6]
                if len(candidates) >= 2:
                    print("🔍 第四轮：VLM候选强制判别...")
                    choice_result = vlm_forced_choice(self.client, image_path, idiom, candidates, self.vision_model)
                    print(f"   候选: {' / '.join(candidates)}")
                    print(f"   选择: {choice_result['chosen']}")
                    print(f"   支持目标: {'✅' if choice_result['supports_target'] else '❌'}")
                    eval_result["forced_choice"] = choice_result

                    # 如果盲猜失败但候选判别强支持目标，可标记为“近成功”
                    if (not eval_result["is_correct"]) and choice_result["supports_target"] and choice_result["confidence"] >= 0.75:
                        eval_result["near_hit"] = True
                else:
                    eval_result["forced_choice"] = {"chosen": "", "confidence": 0.0, "reasoning": "候选不足", "supports_target": False}

            attempt = {
                "iteration": iteration,
                "pun": pun,
                "scene_zh": scene_zh,
                "scene_en": scene_en,
                "image_path": image_path,
                "quality_result": quality_result,
                "vlm_guess": eval_result["guessed"],
                "vlm_confidence": eval_result["confidence"],
                "vlm_reasoning": eval_result["reasoning"],
                "vlm_diagnosis": eval_result.get("diagnosis", {}),
                "vlm_forced_choice": eval_result.get("forced_choice", {}),
                "near_hit": eval_result.get("near_hit", False),
                "is_correct": eval_result["is_correct"],
                "quality_passed": eval_result.get("quality_passed", False),
            }
            reflection = reflect_attempt(self.client, idiom, attempt, self.text_model)
            attempt["reflection"] = reflection
            attempts.append(attempt)
            history.append(attempt)
            print(f"🤔 反思: {reflection.lesson_learned[:100]}...")

            if eval_result["is_correct"] and eval_result["confidence"] >= confidence_threshold:
                print("\n🎉 成功！VLM正确猜出成语")
                save_experience(self.client, self.memory, idiom, attempt, reflection, history[:-1], self.text_model)
                return True, attempts, generate_summary(attempts, True)

            print("⚠️ 未猜中，继续优化...")

        print("\n❌ 达到最大迭代次数，未能成功")
        if attempts:
            save_experience(
                self.client, self.memory, idiom, attempts[-1], attempts[-1]["reflection"], history, self.text_model
            )
        return False, attempts, generate_summary(attempts, False)
