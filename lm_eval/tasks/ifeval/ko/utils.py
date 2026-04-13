"""IFEval 한국어 평가용 utils.

원본 ifeval utils를 재사용하고, 한국어 상수 오버라이드 + 상세 리포트 추가.
"""

import collections

from lm_eval.tasks.ifeval import instructions
from lm_eval.tasks.ifeval.utils import (
    InputExample,
    OutputExample,
    agg_inst_level_acc,
    process_results,
    test_instruction_following_loose,
    test_instruction_following_strict,
)

# 한국어 ConstrainedResponse 옵션 오버라이드
instructions._CONSTRAINED_RESPONSE_OPTIONS = (
    "제 대답은 예 입니다.",
    "제 대답은 아니오 입니다.",
    "제 대답은 아마도 입니다.",
)


def print_report(all_outputs):
    """카테고리별 상세 정확도 리포트 출력."""
    print("=" * 100)
    print("KO_IFEVAL DETAILED RESULTS")
    print("=" * 100)

    for method_name, key in [("loose", "out_loose"), ("strict", "out_strict")]:
        method_results = [op[key] for op in all_outputs]

        prompt_total = 0
        prompt_correct = 0
        instruction_total = 0
        instruction_correct = 0

        tier0_total = collections.defaultdict(int)
        tier0_correct = collections.defaultdict(int)
        tier1_total = collections.defaultdict(int)
        tier1_correct = collections.defaultdict(int)

        for example in method_results:
            follow_instruction_list = example.follow_instruction_list
            instruction_id_list = example.instruction_id_list

            prompt_total += 1
            if all(follow_instruction_list):
                prompt_correct += 1

            instruction_total += len(instruction_id_list)
            instruction_correct += sum(follow_instruction_list)

            for instruction_id, followed in zip(
                instruction_id_list, follow_instruction_list
            ):
                category = instruction_id.split(":")[0]
                tier0_total[category] += 1
                if followed:
                    tier0_correct[category] += 1

                tier1_total[instruction_id] += 1
                if followed:
                    tier1_correct[instruction_id] += 1

        print("-" * 70)
        print(f"METHOD: {method_name} (n={len(method_results)})")
        print(f"  prompt-level:      {prompt_correct / prompt_total:.4f}")
        print(f"  instruction-level: {instruction_correct / instruction_total:.4f}")

        print("\n  [Category]")
        for cat in sorted(tier0_total.keys()):
            acc = tier0_correct[cat] / tier0_total[cat]
            print(f"    {cat}: {acc:.4f}")

        print("\n  [Instruction]")
        for inst in sorted(tier1_total.keys()):
            acc = tier1_correct[inst] / tier1_total[inst]
            print(f"    {inst}: {acc:.4f}")

        print("=" * 100)
