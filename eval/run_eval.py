import sys
import os
import json

# Allow importing from src/ regardless of where this script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_agent import ask

EVAL_FILE = os.path.join(os.path.dirname(__file__), "seed_eval.json")


def load_eval_set():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def check_scenario(scenario: dict, result: dict) -> tuple[bool, str]:
    """
    Returns (passed, reason) for a single scenario.
    """
    expect_refusal = scenario["expect_refusal"]
    got_sources = result.get("sources", [])
    answer_text = result.get("answer", "").lower()

    # Heuristic: treat it as a refusal if sources are empty and the answer contains typical refusal language.
    looks_like_refusal = (
        len(got_sources) == 0
        or "don't have information" in answer_text
        or "contact the it helpdesk" in answer_text
    )

    if expect_refusal:
        if looks_like_refusal:
            return True, "Correctly refused."
        else:
            return False, f"Should have refused but answered using sources: {got_sources}"
    else:
        if looks_like_refusal:
            return False, "Incorrectly refused when an answer was expected."
        expected_sources = set(scenario.get("expected_sources", []))
        actual_sources = set(got_sources)
        if expected_sources and not (expected_sources & actual_sources):
            return False, (
                f"Answered, but cited sources don't match expected. "
                f"Expected one of {expected_sources}, got {actual_sources}"
            )
        return True, f"Answered with sources: {got_sources}"


def run_eval():
    scenarios = load_eval_set()
    results = []
    passed_count = 0

    for scenario in scenarios:
        print(f"\n[{scenario['id']}] {scenario['question']}")
        result = ask(scenario["question"])
        passed, reason = check_scenario(scenario, result)

        status = "PASS" if passed else "FAIL"
        print(f"  -> {status}: {reason}")
        print(f"  -> Answer: {result.get('answer', '')[:150]}...")

        if passed:
            passed_count += 1

        results.append({
            "id": scenario["id"],
            "question": scenario["question"],
            "passed": passed,
            "reason": reason,
        })

    total = len(scenarios)
    accuracy = (passed_count / total) * 100

    print("\n" + "=" * 50)
    print(f"RESULTS: {passed_count}/{total} passed ({accuracy:.1f}%)")
    print("=" * 50)

    if accuracy >= 85:
        print("✅ Meets the 85% checkpoint target.")
    else:
        print("⚠️  Below 85% target. Review failures above and tune:")
        print("   - MAX_DISTANCE_THRESHOLD in knowledge_agent.py")
        print("   - chunking strategy in ingest.py")
        print("   - system prompt wording")

    return results, accuracy


if __name__ == "__main__":
    run_eval()