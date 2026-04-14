import base64
import json
import os
from typing import Dict

import requests

from agent_utils import parse_bool
from prompts import (
    VLM_GUESS_PROMPT,
    build_quality_eval_prompt,
    build_vlm_diagnosis_prompt,
    build_vlm_forced_choice_prompt,
)


def generate_image(client, scene_en: str, idiom: str, iteration: int) -> str:
    """生成图片 - 使用z-image-turbo"""
    try:
        style = "Bright cartoon style, clean background, simple composition, cute and expressive"
        prompt = f"{scene_en}. {style}"

        print("   调用z-image-turbo生成图片...")
        print(f"   提示词: {prompt[:80]}...")

        response = client.images.generate(model="z-image-turbo", prompt=prompt, size="1024x1024", n=1)
        image_url = response.data[0].url
        if not image_url:
            print("   ❌ API返回空URL")
            return ""

        print("   图片URL获取成功")
        image_data = requests.get(image_url, timeout=60).content
        os.makedirs("output/images", exist_ok=True)
        image_path = f"output/images/{idiom}_v{iteration}.png"
        with open(image_path, "wb") as f:
            f.write(image_data)
        print(f"   ✅ 图片已保存: {image_path}")
        return image_path
    except Exception as e:
        print(f"❌ 图片生成失败: {e}")
        return ""


def evaluate_image_quality(client, image_path: str, scene_zh: str, pun: str, idiom: str, vision_model: str) -> Dict:
    """第一轮：评估图片质量，检查是否符合场景描述"""
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        system_prompt = build_quality_eval_prompt(scene_zh, pun, idiom)

        response = client.chat.completions.create(
            model=vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：评估图片是否准确表达目标场景与双关语（目标成语：{idiom}，双关语：{pun}）。"
                                "请按要求打分并输出JSON。"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                },
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        overall = float(result.get("overall_score", 0))
        interference = float(result.get("interference_score", 0))
        text_found = parse_bool(result.get("text_found", False))
        passed = False if text_found else overall >= 0.7 and interference >= 0.8
        if text_found:
            print("   🚫 检测到图片中出现文字，判定不通过！")

        return {
            "overall_score": overall,
            "subject_score": float(result.get("subject_score", 0)),
            "action_score": float(result.get("action_score", 0)),
            "scene_score": float(result.get("scene_score", 0)),
            "interference_score": interference,
            "text_found": text_found,
            "passed": passed,
            "analysis": result.get("analysis", ""),
        }
    except Exception as e:
        print(f"❌ 图片质量评估失败: {e}")
        return {"passed": False, "overall_score": 0, "analysis": f"评估失败: {e}"}


def vlm_evaluate(client, image_path: str, idiom: str, vision_model: str) -> Dict:
    """第二轮：VLM看图猜成语"""
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        system_prompt = VLM_GUESS_PROMPT
        response = client.chat.completions.create(
            model=vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：你是盲猜模式，只看图猜原成语（目标任务成语：{idiom}，但不要直接复述该词）。"
                                "请给出最可能成语、信心度和推理，并输出JSON。"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                },
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        guessed = result.get("guessed", "").strip()
        return {
            "guessed": guessed,
            "confidence": float(result.get("confidence", 0)),
            "reasoning": result.get("reasoning", ""),
            "is_correct": idiom in guessed or guessed in idiom,
        }
    except Exception as e:
        print(f"❌ VLM评估失败: {e}")
        return {"guessed": "", "confidence": 0.0, "reasoning": f"评估出错: {str(e)}", "is_correct": False}


def vlm_diagnose_with_answer(client, image_path: str, idiom: str, pun: str, scene_zh: str, vision_model: str) -> Dict:
    """第三轮：告知正确答案后，让VLM独立诊断图片表达问题"""
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        system_prompt = build_vlm_diagnosis_prompt(idiom, pun, scene_zh)
        response = client.chat.completions.create(
            model=vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：带答案诊断模式。已知正确成语是“{idiom}”，请分析该图为何容易/不容易被猜中。"
                                "请输出结构化JSON诊断结果。"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                },
            ],
            temperature=0.2,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "subject_clarity": float(result.get("subject_clarity", 0)),
            "visual_readability": float(result.get("visual_readability", 0)),
            "pun_mapping": float(result.get("pun_mapping", 0)),
            "misleading_risk": float(result.get("misleading_risk", 0)),
            "analysis": result.get("analysis", ""),
            "key_issues": result.get("key_issues", []),
            "suggestions": result.get("suggestions", []),
        }
    except Exception as e:
        print(f"❌ VLM带答案诊断失败: {e}")
        return {
            "subject_clarity": 0.0,
            "visual_readability": 0.0,
            "pun_mapping": 0.0,
            "misleading_risk": 1.0,
            "analysis": f"诊断失败: {str(e)}",
            "key_issues": [],
            "suggestions": ["继续使用盲猜与质量评估进行迭代"],
        }


def vlm_forced_choice(client, image_path: str, target: str, candidates: list, vision_model: str) -> Dict:
    """第四轮：给定候选列表强制选择，判断是否更支持目标词"""
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = build_vlm_forced_choice_prompt(target, candidates)
        response = client.chat.completions.create(
            model=vision_model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "任务：请在候选中强制选择最匹配项，并返回JSON。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        chosen = result.get("chosen", "")
        supports_target = parse_bool(result.get("supports_target", False)) or (chosen == target)
        return {
            "chosen": chosen,
            "confidence": float(result.get("confidence", 0)),
            "reasoning": result.get("reasoning", ""),
            "supports_target": supports_target,
        }
    except Exception as e:
        print(f"❌ VLM候选判别失败: {e}")
        return {"chosen": "", "confidence": 0.0, "reasoning": f"候选判别失败: {str(e)}", "supports_target": False}
