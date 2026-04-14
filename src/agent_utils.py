from typing import Set, Tuple

from pypinyin import Style, pinyin


def _get_pronunciations(ch: str) -> Set[str]:
    """获取单字可能拼音（不含声调），用于同音校验。"""
    values = pinyin(ch, style=Style.NORMAL, heteronym=True, strict=False)
    if not values or not values[0]:
        return set()
    return {item.lower() for item in values[0] if item}


def check_pun_valid(idiom: str, pun: str) -> Tuple[bool, str]:
    """
    检查双关语是否有效
    要求：
    1) 至少替换1个字
    2) 长度相同
    3) 每个替换位必须同音（忽略声调）
    """
    pun_clean = pun.split("（")[0].strip()

    if len(idiom) != len(pun_clean):
        return False, f"长度不一致: {idiom}({len(idiom)}) vs {pun_clean}({len(pun_clean)})"

    diff_count = sum(1 for c1, c2 in zip(idiom, pun_clean) if c1 != c2)

    if diff_count == 0:
        return False, "没有替换任何字（完全相同）"

    mismatch_details = []
    for idx, (src, dst) in enumerate(zip(idiom, pun_clean), start=1):
        if src == dst:
            continue
        src_py = _get_pronunciations(src)
        dst_py = _get_pronunciations(dst)
        if not src_py or not dst_py or src_py.isdisjoint(dst_py):
            mismatch_details.append(f"第{idx}字 '{src}'→'{dst}' 非同音")

    if mismatch_details:
        return False, "；".join(mismatch_details)

    return True, f"有效双关: 替换了{diff_count}个字"

def parse_bool(value) -> bool:
    """兼容处理LLM返回的布尔值（bool/字符串/数字）"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是"}
    return False
