import json
from typing import Dict, List, Optional, Tuple

from agent_utils import check_pun_valid
from agent_types import ReflectionResult
from prompts import build_initial_generation_prompt, build_regeneration_prompt


def load_examples_text() -> str:
    """加载示例文件作为参考"""
    try:
        with open("examples/chengyu_examples.md", "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        examples = []
        for line in lines:
            if line.startswith("|") and "原词" not in line and "---" not in line:
                parts = [p.strip() for p in line.split("|")]
                parts = [p for p in parts if p]
                if len(parts) >= 3:
                    examples.append({"original": parts[0], "pun": parts[1], "scene": parts[2]})

        if examples:
            output = ["\n## 参考示例\n"]
            for e in examples[:5]:
                output.append(f"- {e['original']} → {e['pun']}: {e['scene'][:50]}...")
            return "\n".join(output)
    except Exception:
        pass
    return ""


def build_memory_context(memory, history: List[Dict]) -> str:
    """构建记忆上下文（反思驱动，避免日志噪声）"""
    context_parts = [memory.get_rules_text(), memory.get_success_patterns()]
    examples_text = load_examples_text()
    if examples_text:
        context_parts.append(examples_text)

    if history:
        # 只保留反思结论，不传原始错误流水
        recent_reflections = []
        for h in reversed(history):
            reflection = h.get("reflection")
            if reflection:
                recent_reflections.append((h.get("iteration"), reflection))
            if len(recent_reflections) >= 3:
                break

        if recent_reflections:
            context_parts.append("\n## 最近反思要点（用于下一轮改进）\n")
            for iteration, reflection in reversed(recent_reflections):
                analysis = (reflection.analysis or "").strip()
                suggestions = [s.strip() for s in (reflection.suggestions or []) if isinstance(s, str) and s.strip()]
                key_factors = [k.strip() for k in (reflection.key_factors or []) if isinstance(k, str) and k.strip()]

                if analysis:
                    context_parts.append(f"- 第{iteration}轮分析: {analysis[:120]}")
                if key_factors:
                    context_parts.append(f"  关键因素: {'; '.join(key_factors[:3])}")
                if suggestions:
                    context_parts.append(f"  改进建议: {'; '.join(suggestions[:3])}")

    return "\n".join(context_parts)


def _pick_valid_pun(idiom: str, result: Dict, fallback_pun: str = "") -> str:
    """从模型结果中选择首个通过硬校验的候选双关语。"""
    direct_pun = result.get("pun", "") or fallback_pun
    is_valid, _ = check_pun_valid(idiom, direct_pun) if direct_pun else (False, "")
    if is_valid:
        return direct_pun

    for item in result.get("candidates", []) or []:
        candidate_pun = item.get("pun", "")
        ok, _ = check_pun_valid(idiom, candidate_pun) if candidate_pun else (False, "")
        if ok:
            return candidate_pun

    return direct_pun


def generate_initial(client, idiom: str, memory_context: str, text_model: str) -> Optional[Tuple]:
    system_prompt = build_initial_generation_prompt(memory_context)
    user_prompt = f"成语：{idiom}\n\n请生成双关语和场景描述。"
    try:
        response = client.chat.completions.create(
            model=text_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.8,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        chosen_pun = _pick_valid_pun(idiom, result)
        return chosen_pun, result.get("scene_zh", ""), result.get("scene_en", ""), result.get("reasoning", "")
    except Exception as e:
        print(f"生成错误: {e}")
        return None


def regenerate_with_lessons(
    client,
    idiom: str,
    prev_attempt: Dict,
    reflection: ReflectionResult,
    memory_context: str,
    text_model: str,
) -> Optional[Tuple]:
    system_prompt = build_regeneration_prompt(memory_context, prev_attempt, reflection)
    user_prompt = f"成语：{idiom}\n\n请根据反思改进生成。"
    try:
        response = client.chat.completions.create(
            model=text_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        chosen_pun = _pick_valid_pun(idiom, result, fallback_pun=prev_attempt["pun"])
        return (
            chosen_pun,
            result.get("scene_zh", ""),
            result.get("scene_en", ""),
            result.get("reasoning", "") + f" [改进: {result.get('changes', '')}]",
        )
    except Exception as e:
        print(f"重新生成错误: {e}")
        return None
