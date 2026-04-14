import json
from datetime import datetime
from typing import Dict, List

from agent_types import ReflectionResult
from memory import Experience
from prompts import (
    build_reflection_prompt_for_guess_failure,
    build_reflection_prompt_for_guess_success,
    build_reflection_prompt_for_quality_failure,
    build_rule_extraction_prompt,
)


def _normalize_to_str_list(value) -> List[str]:
    """将LLM返回的列表统一转成字符串列表，防止出现dict等结构导致拼接崩溃。"""
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            # 常见结构如 {"factor": "..."} 或 {"issue": "..."}
            text = item.get("factor") or item.get("issue") or item.get("text") or json.dumps(item, ensure_ascii=False)
            normalized.append(str(text))
        else:
            normalized.append(str(item))
    return normalized


def reflect_attempt(client, idiom: str, current: Dict, text_model: str) -> ReflectionResult:
    is_correct = current["is_correct"]
    quality_passed = current.get("quality_passed", True)
    quality_result = current.get("quality_result", {})

    if not quality_passed:
        system_prompt = build_reflection_prompt_for_quality_failure(quality_result, current["scene_zh"])
    else:
        diagnosis = current.get("vlm_diagnosis", {})
        if is_correct:
            system_prompt = build_reflection_prompt_for_guess_success(idiom, current, diagnosis)
        else:
            system_prompt = build_reflection_prompt_for_guess_failure(idiom, current, diagnosis)
            forced = current.get("vlm_forced_choice", {})
            if forced:
                system_prompt += (
                    f"\n补充信息：候选强制判别选择={forced.get('chosen', '')}，"
                    f"支持目标={forced.get('supports_target', False)}，"
                    f"信心度={forced.get('confidence', 0):.0%}，"
                    f"理由={forced.get('reasoning', '')}"
                )

    try:
        response = client.chat.completions.create(
            model=text_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "请进行反思分析。"}],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return ReflectionResult(
            success=is_correct,
            analysis=result.get("analysis", ""),
            key_factors=_normalize_to_str_list(result.get("key_factors", [])),
            suggestions=_normalize_to_str_list(result.get("suggestions", [])),
            lesson_learned=result.get("lesson_learned", ""),
        )
    except Exception as e:
        print(f"反思错误: {e}")
        return ReflectionResult(success=is_correct, analysis="反思失败", key_factors=[], suggestions=["重新尝试"], lesson_learned="需要更多尝试")


def is_new_insight(reflection: ReflectionResult, history: List[Dict]) -> bool:
    if len(history) == 0:
        return True
    prev_factors = set()
    for h in history:
        if "reflection" in h:
            prev_factors.update(h["reflection"].key_factors)
    current_factors = set(reflection.key_factors)
    new_factors = current_factors - prev_factors
    if new_factors:
        print(f"💡 发现新问题: {new_factors}")
        return True
    print(f"⏭️ 问题与之前重复: {current_factors}")
    return False


def extract_rules_from_reflection(client, memory, reflection: ReflectionResult, success: bool, text_model: str):
    existing_rules = set(memory.rules.keys())
    system_prompt = build_rule_extraction_prompt(existing_rules)
    user_prompt = f"""本次生成：{'成功' if success else '失败'}
反思分析：{reflection.analysis}
关键因素：{', '.join(reflection.key_factors)}
改进建议：{', '.join(reflection.suggestions)}"""
    try:
        response = client.chat.completions.create(
            model=text_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        new_rules = json.loads(response.choices[0].message.content)
        truly_new_rules = {k: v for k, v in new_rules.items() if k not in existing_rules}
        if truly_new_rules:
            memory.update_rules(truly_new_rules)
            print(f"💡 LLM总结出 {len(truly_new_rules)} 条新规则: {list(truly_new_rules.keys())}")
        else:
            print("💡 没有新规则需要添加（已被现有规则覆盖）")
    except Exception as e:
        print(f"规则提取失败: {e}")


def save_experience(client, memory, idiom: str, attempt: Dict, reflection: ReflectionResult, history: List[Dict], text_model: str):
    is_improvement = False
    improvement_reason = ""

    if len(history) == 0:
        is_improvement = True
        improvement_reason = "第一轮尝试"
    else:
        prev = history[-1]
        if attempt["is_correct"] and not prev["is_correct"]:
            is_improvement = True
            improvement_reason = "成功猜对"
        elif attempt["is_correct"] and prev["is_correct"] and attempt["vlm_confidence"] > prev["vlm_confidence"]:
            is_improvement = True
            improvement_reason = f"信心度提升 {prev['vlm_confidence']:.0%} → {attempt['vlm_confidence']:.0%}"
        elif not attempt["is_correct"] and not prev["is_correct"]:
            curr_quality = attempt.get("quality_result", {}).get("overall_score", 0)
            prev_quality = prev.get("quality_result", {}).get("overall_score", 0)
            if curr_quality > prev_quality:
                is_improvement = True
                improvement_reason = f"图片质量提升 {prev_quality:.0%} → {curr_quality:.0%}"
            elif idiom in attempt["vlm_guess"] and idiom not in prev["vlm_guess"]:
                is_improvement = True
                improvement_reason = "猜测更接近目标"
            elif attempt["vlm_confidence"] < prev["vlm_confidence"] and curr_quality <= prev_quality:
                is_improvement = False
                improvement_reason = "信心度下降且无质量提升，跳过保存"
            else:
                is_improvement = True
                improvement_reason = f"第{attempt['iteration']}轮尝试（累积经验）"

    if not is_improvement:
        print(f"⏭️ 跳过经验保存: {improvement_reason}")
        return

    q = attempt.get("quality_result", {})
    quality_info = f"[质量: {q.get('overall_score', 0):.0%}, 通过: {q.get('passed', False)}] "
    exp = Experience(
        idiom=idiom,
        pun=attempt["pun"],
        scene_zh=attempt["scene_zh"],
        success=attempt["is_correct"],
        iteration=attempt["iteration"],
        reason=f"{quality_info}{reflection.analysis} [{improvement_reason}]",
        key_factors=reflection.key_factors,
        timestamp=datetime.now().isoformat(),
        vlm_feedback=attempt["vlm_reasoning"],
    )
    memory.add_experience(exp)
    if attempt["is_correct"] or is_new_insight(reflection, history):
        extract_rules_from_reflection(client, memory, reflection, attempt["is_correct"], text_model)


def generate_summary(attempts: List[Dict], success: bool) -> str:
    if not attempts:
        return "未能完成任何迭代。"
    total = len(attempts)
    final = attempts[-1]
    quality = final.get("quality_result", {})
    quality_str = f"(图片质量: {quality.get('overall_score', 0):.0%})"
    if success:
        return f"成语'{final['pun']}'经过{total}轮迭代成功生成可识别的视觉双关语。{quality_str}"
    if not final.get("quality_passed", True):
        return f"成语经过{total}轮迭代，最后一轮图片质量检查未通过{quality_str}，建议优化场景描述。"
    return f"成语经过{total}轮迭代未能成功{quality_str}，建议更换双关思路或人工优化。"
