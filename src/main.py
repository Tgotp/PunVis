"""
主程序入口
"""

import argparse
import os
import shutil
import sys
from openai import OpenAI

from memory import ExperienceMemory
from agent import PunVisAgent


def setup_console_encoding():
    """在 Windows 控制台下启用 UTF-8，避免 emoji 打印报错"""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        # 编码设置失败不影响主流程
        pass


def cleanup_images(output_dir="output/images"):
    """清理之前生成的图片"""
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(output_dir, f))
        print(f"🧹 已清理 {output_dir} 中的旧图片")


def main():
    setup_console_encoding()

    # 先加载配置
    try:
        from config import (
            OPENAI_API_KEY, OPENAI_BASE_URL,
            MAX_ITERATIONS, MIN_CONFIDENCE,
            TEXT_MODEL, VISION_MODEL
        )
    except ImportError:
        print("❌ 请先创建 src/config.py 文件")
        return

    parser = argparse.ArgumentParser(description='成语双关语可视化Agent')
    parser.add_argument('--idiom', type=str, required=True, help='要处理的成语')
    parser.add_argument('--max-iterations', type=int, default=MAX_ITERATIONS, help=f'最大迭代次数 (config默认: {MAX_ITERATIONS})')
    parser.add_argument('--min-confidence', type=float, default=MIN_CONFIDENCE, help=f'最低信心度 (config默认: {MIN_CONFIDENCE})')
    args = parser.parse_args()

    # 清理旧图片
    cleanup_images()

    print(
        f"⚙️ 使用配置: max_iterations={args.max_iterations}, "
        f"min_confidence={args.min_confidence}, text_model={TEXT_MODEL}, vision_model={VISION_MODEL}"
    )

    # 创建客户端
    client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    client = OpenAI(**client_kwargs)

    # 初始化记忆系统
    memory = ExperienceMemory(project_root=".")

    # 创建Agent
    agent = PunVisAgent(client, memory, text_model=TEXT_MODEL, vision_model=VISION_MODEL)

    # 运行
    success, attempts, summary = agent.generate_with_reflection(
        args.idiom,
        max_iterations=args.max_iterations,
        confidence_threshold=args.min_confidence
    )

    # 打印最终结果
    print(f"\n{'='*60}")
    print("📊 最终结果")
    print(f"{'='*60}")
    print(f"成语: {args.idiom}")
    print(f"成功: {'✅' if success else '❌'}")
    print(f"迭代次数: {len(attempts)}")

    if attempts:
        final = attempts[-1]
        print(f"最终双关语: {final['pun']}")
        print(f"最终场景: {final['scene_zh']}")

        # 显示质量评估信息
        if 'quality_result' in final:
            q = final['quality_result']
            print(f"\n📷 图片质量评估:")
            print(f"   总体得分: {q.get('overall_score', 0):.0%}")
            print(f"   主体呈现: {q.get('subject_score', 0):.0%}")
            print(f"   动作表达: {q.get('action_score', 0):.0%}")
            print(f"   场景匹配: {q.get('scene_score', 0):.0%}")
            print(f"   干扰元素: {q.get('interference_score', 0):.0%}")
            if q.get('text_found', False):
                print(f"   🚫 文字检测: 发现文字！")
            else:
                print(f"   ✅ 文字检测: 通过")
            print(f"   质量通过: {'✅' if q.get('passed', False) else '❌'}")

        if final.get('quality_passed', True):
            print(f"\n🎯 VLM猜测: {final['vlm_guess']}")
            print(f"   信心度: {final['vlm_confidence']:.0%}")

    print(f"\n总结: {summary}")

    # 打印记忆统计
    stats = memory.get_statistics()
    print(f"\n📚 Agent记忆统计:")
    print(f"  - 总经验: {stats['total']}")
    print(f"  - 成功率: {stats['success_rate']:.1%}")
    print(f"  - 平均迭代: {stats['avg_iterations']:.1f}轮")


if __name__ == "__main__":
    main()
