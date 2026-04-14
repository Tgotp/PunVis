"""集中管理所有 LLM Prompt 文本。"""


def build_initial_generation_prompt(memory_context: str) -> str:
    return f"""你是一位精通中文双关语的创意专家。请将成语转换为同音双关语，并生成场景描述。

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
4. **禁止非同音替换**：任一替换位若非同音，不可作为最终结果

## 生成流程（必须执行）
1) 先提出 4~5 个双关候选（每个都给替换计划与拼音）
2) 对每个候选执行同音自检（逐替换位）
3) 只从自检通过的候选中选择一个“最佳候选”
4) 基于最佳候选生成场景描述

## 选择规则
- 优先选择：同音替换自然、语义可视化强、场景容易理解的候选
- 若多个候选都合格，选视觉表达最清晰的一项

## 场景要求
- 只用纯视觉元素（动作、物体、表情）表达双关含义
- 通过动作和物体关系表达含义，而非文字

## 参考示例风格
- 鸡不可湿 → 小鸡撑伞避雨（动作表达）
- 守猪待兔 → 农夫抱猪等兔子
- 对牛谈情 → 人向牛表白，牛害羞

输出JSON格式：
{{
    "candidates": [
        {{
            "pun": "候选双关语",
            "replacement_plan": [
                {{
                    "index": 1,
                    "source_char": "原字",
                    "target_char": "替换字",
                    "source_pinyin": "拼音",
                    "target_pinyin": "拼音"
                }}
            ],
            "phonetic_pass": true,
            "phonetic_notes": "同音检查说明"
        }}
    ],
    "pun": "最终选择的双关语",
    "scene_zh": "中文场景描述",
    "scene_en": "英文场景描述",
    "reasoning": "生成思路说明（说明为何从候选中选择该项）"
}}"""


def build_regeneration_prompt(memory_context: str, prev_attempt: dict, reflection) -> str:
    return f"""你是一位精通中文双关语的创意专家。上一轮生成失败了，请根据反思和已总结的规则改进。

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

## 双关语硬约束
1. 必须至少替换1个字（禁止与原成语完全相同）
2. 替换字必须同音或近同音
3. 字数必须与原成语一致
4. 如果输出与原成语相同，视为失败
5. 若任一替换位非同音，不可作为最终结果

## 非同音防错流程
1) 先列出 4~5 个候选（每个含替换计划与拼音）
2) 对每个候选逐位同音自检
3) 从合格候选中选择最佳并生成场景
4) 如果都不完美，仍需返回一个最优候选并说明取舍

请根据以上信息，生成改进后的双关语和场景。

输出JSON格式：
{{
    "candidates": [
        {{
            "pun": "候选双关语",
            "replacement_plan": [
                {{
                    "index": 1,
                    "source_char": "原字",
                    "target_char": "替换字",
                    "source_pinyin": "拼音",
                    "target_pinyin": "拼音"
                }}
            ],
            "phonetic_pass": true,
            "phonetic_notes": "同音检查说明"
        }}
    ],
    "pun": "最终选择的双关语",
    "scene_zh": "改进后的中文场景描述（禁止任何文字、标牌、字母）",
    "scene_en": "改进后的英文场景描述",
    "reasoning": "改进思路说明",
    "changes": "具体改了什么"
}}"""


def build_quality_eval_prompt(scene_zh: str, pun: str, idiom: str) -> str:
    return f"""你是一位图片质量评估专家。请严格评估生成的图片是否符合预期的场景描述。
## 🚨 最高优先级检查（一票否决）
**图片中绝对不能出现任何文字、字母、标牌、符号、书法、刻字！**
## 场景描述
{scene_zh}
## 预期双关语
{pun} (原成语: {idiom})
## 输出格式
{{"overall_score":"0-1","subject_score":"0-1","action_score":"0-1","scene_score":"0-1","interference_score":"0-1","text_found":"true/false","passed":"bool","analysis":"详细分析"}}"""


VLM_GUESS_PROMPT = """你是“成语视觉双关推理专家”，正在参与一个“成语视觉双关”任务，流程如下：
1) 先把原成语中的某个字替换成同音字，得到双关语
2) 再根据双关语设计一个可视化场景（主体、动作、对比、环境）
3) 再把该场景生成图片
4) 你现在的职责：只看这张图片，反推原成语

关键说明：
- 图片通常呈现的是“双关语对应的画面”，不是原成语字面画面
- 画面中的每个关键元素（主体/动作/配角/道具/环境）都有语义作用，请逐一判断其作用
- 你需要把画面元素做同音映射，再回推原成语
- 若存在多个候选，请给最可能的一个并说明为什么

## 强制推理顺序（必须执行）
1) 先识别“字级锚点”：画面中最可能对应双关替换字的元素（例如碗/墙/曲）
2) 再做“同音回推”：把锚点字映射回原词可能字（例如碗→顽，墙→强，曲→屈）
3) 最后才给出原成语

## 约束
- 禁止直接根据抽象语义（如“坚韧”“努力”）跳到常见成语
- 若锚点不足或映射不完整，必须降低置信度

输出JSON：
{
  "anchors": ["识别到的关键锚点元素"],
  "char_mapping": [
    {"anchor":"锚点元素", "pun_char":"双关字", "target_char":"回推原字", "confidence":0-1}
  ],
  "guessed":"原成语",
  "confidence":0-1,
  "reasoning":"先锚点、后字级映射、再结论"
}"""


def build_vlm_diagnosis_prompt(idiom: str, pun: str, scene_zh: str) -> str:
    return f"""你是“成语视觉双关诊断专家”，正在参与一个“成语视觉双关”任务复盘环节。
任务机制：原成语 -> 同音双关语 -> 场景设计 -> 生成图片 -> VLM盲猜原成语。
你现在处于“带答案诊断”阶段，需要判断这张图为什么容易/不容易被猜回原成语。

已知答案如下：
- 原成语: {idiom}
- 双关语: {pun}
- 目标场景: {scene_zh}
请特别检查：场景中的每个关键元素是否都在为“猜回原成语”服务，还是在制造歧义。
并重点判断：观察者是否能从“元素 -> 双关字 -> 原字”完成闭环。

输出JSON:
{{
  "subject_clarity":0-1,
  "visual_readability":0-1,
  "pun_mapping":0-1,
  "misleading_risk":0-1,
  "anchor_detection":0-1,
  "char_level_recoverability":0-1,
  "analysis":"结论",
  "key_issues":["问题1","问题2"],
  "suggestions":["建议1","建议2"]
}}"""


def build_vlm_forced_choice_prompt(target: str, candidates: list) -> str:
    candidate_text = "、".join(candidates)
    return f"""你是“成语视觉双关判别专家”。
你将看到一张图，并在给定候选中必须选择最匹配的一项。

任务背景：
- 这是同音双关生成任务，图片可能是双关语场景，不是原词字面场景。
- 你需要基于画面元素与语义映射，选择最可能对应的目标词。

目标参考（不保证一定正确，仅供比对）：{target}
候选列表：{candidate_text}

输出JSON：
{{"chosen":"从候选中选一个","confidence":0-1,"reasoning":"为什么选这个","supports_target":true/false}}"""


def build_reflection_prompt_for_quality_failure(quality_result: dict, scene_zh: str) -> str:
    return f"""你是一位图片质量分析专家。图片未能通过质量检查，请分析问题所在。
质量检查结果：总体{quality_result.get('overall_score', 0):.0%}，主体{quality_result.get('subject_score', 0):.0%}，动作{quality_result.get('action_score', 0):.0%}，场景{quality_result.get('scene_score', 0):.0%}，干扰{quality_result.get('interference_score', 0):.0%}。
场景描述：{scene_zh}
输出JSON：{{"analysis":"","key_factors":[],"suggestions":[],"lesson_learned":""}}"""


def build_reflection_prompt_for_guess_failure(idiom: str, current: dict, diagnosis: dict) -> str:
    return f"""你是一位成语双关语创作批评专家。请分析为什么VLM没能猜出正确成语。
原成语: {idiom}
双关语: {current['pun']}
场景: {current['scene_zh']}
VLM猜测: {current['vlm_guess']}
VLM推理: {current['vlm_reasoning']}
带答案诊断: {diagnosis.get('analysis', '无')}

请按“可执行归因”输出，不要空泛总结：
1) 锚点识别失败了吗？（是/否 + 证据）
2) 是否被抽象语义吸走（如坚韧/努力）而忽略字级映射？（是/否 + 证据）
3) 哪个元素没有形成“元素->双关字->原字”闭环？
4) 下一轮必须删除哪些误导元素？必须强化哪些锚点元素？

输出JSON：{{"analysis":"","key_factors":[],"suggestions":[],"lesson_learned":""}}"""


def build_reflection_prompt_for_guess_success(idiom: str, current: dict, diagnosis: dict) -> str:
    return f"""你是一位成语视觉双关复盘专家。VLM已经猜对，请总结“为什么成功”以及可复用经验。
原成语: {idiom}
双关语: {current['pun']}
场景: {current['scene_zh']}
VLM猜测: {current['vlm_guess']}
VLM推理: {current['vlm_reasoning']}
带答案诊断: {diagnosis.get('analysis', '无')}

请聚焦成功因素：
1) 哪些场景元素有效支撑了从双关语回推原成语
2) 哪些对比或动作提高了可读性
3) 该成功策略可如何复用到其他成语

输出JSON：{{"analysis":"成功原因总结","key_factors":[],"suggestions":[],"lesson_learned":"一句话成功经验"}}"""


def build_rule_extraction_prompt(existing_rules: set) -> str:
    return f"""你是一位规则总结专家。当前已有规则：{', '.join(existing_rules)}。
从反思中提炼最多1-2条新规则。若无新规则返回{{}}。仅输出JSON对象。"""
