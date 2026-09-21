"""
The evaluation harness: runs every case in eval_questions.py and reports
what happened.

Run it with:

    python evaluate.py

It does two jobs.

First, it checks behaviour. For each case it compares what the assistant
actually did against what eval_questions.py said it should do - which
tools it chose, whether it grounded or refused, and whether the answer
contains (or avoids) the expected wording.

Second, it produces the numbers needed to set MIN_RELEVANCE_SCORE. The
threshold should sit in the gap between the lowest-scoring question that
SHOULD be answered and the highest-scoring question that should NOT be.
The summary at the end prints exactly those two numbers and suggests the
midpoint. If they overlap, no threshold works and the real fix is better
chunking or clearer documents - so the summary says that too, rather
than quietly recommending a number that cannot work.
"""

from agent import AgentReply, Assistant
from eval_questions import EVAL_CASES, EvalCase
import config


def check_case(case: EvalCase, reply: AgentReply) -> list[str]:
    """Compare one reply against what the case expected.

    Args:
        case: The test case, including its expected behaviour.
        reply: What the assistant actually produced.

    Returns:
        A list of failure descriptions. Empty means the case passed.
    """
    failures: list[str] = []
    answer_lower = reply.answer.lower()

    # Check 1: routing. Did it choose the right tools? Compared as sets
    # because the order the model calls them in is its own business -
    # what matters is that it reached for the right ones.
    expected_tools = set(case.expected_tools)
    actual_tools = set(reply.tools_called)
    if expected_tools != actual_tools:
        failures.append(
            f"tools: expected {sorted(expected_tools) or ['none']}, "
            f"got {sorted(actual_tools) or ['none']}"
        )

    # Check 2: grounding. "grounded" answers must have retrieved
    # something or called the ticket tool; "refusal" answers must not be
    # presenting document sources as if they backed an answer.
    if case.expect_answer == "grounded":
        used_ticket = "get_ticket_status" in actual_tools
        if not reply.sources and not used_ticket:
            failures.append("expected a grounded answer but nothing was retrieved")
    elif case.expect_answer == "conversational":
        if reply.tools_called:
            failures.append("expected no tool calls")

    # Check 3: wording. The softest of the three, since the model is free
    # to paraphrase - which is why eval_questions.py only lists
    # distinctive tokens here. A failure is a prompt to go and read the
    # answer, not proof of a bug.
    for phrase in case.must_mention:
        if phrase.lower() not in answer_lower:
            failures.append(f"missing expected text: {phrase!r}")

    for phrase in case.must_not_mention:
        if phrase.lower() in answer_lower:
            failures.append(f"contains forbidden text: {phrase!r}")

    return failures


def run_case(case: EvalCase) -> AgentReply:
    """Run one case end to end and return the final reply.

    A fresh Assistant per case, so cases cannot contaminate each other.
    Multi-turn cases run every turn in order; only the reply to the last
    turn is checked, because that is the one the expectations describe.

    Args:
        case: The case to run.

    Returns:
        The AgentReply from the final turn.
    """
    assistant = Assistant()

    reply = None
    for turn in case.turns:
        reply = assistant.ask(turn)

    return reply


def main() -> None:
    """Run the whole evaluation set and print the results."""
    try:
        config.validate_config()
    except RuntimeError as error:
        print(f"\nConfiguration problem:\n\n{error}\n")
        return

    print(f"\nRunning {len(EVAL_CASES)} cases ")
    print(f"(threshold currently {config.MIN_RELEVANCE_SCORE}, top_k {config.SEARCH_TOP_K})\n")

    passed = 0
    # Collected for the threshold analysis at the end. Only cases that
    # actually searched the documents are relevant - a pure ticket
    # lookup has no retrieval score to contribute.
    should_retrieve_scores: list[tuple[str, float]] = []
    should_not_retrieve_scores: list[tuple[str, float]] = []

    for case in EVAL_CASES:
        try:
            reply = run_case(case)
        except Exception as error:
            print(f"{case.id:<4} ERROR   {type(error).__name__}: {error}")
            continue

        failures = check_case(case, reply)

        if "search_documents" in reply.tools_called:
            if case.expect_answer == "refusal":
                should_not_retrieve_scores.append((case.id, reply.top_score))
            else:
                should_retrieve_scores.append((case.id, reply.top_score))

        status = "PASS" if not failures else "FAIL"
        if not failures:
            passed += 1

        print(f"{case.id:<4} {status}    top score {reply.top_score:.2f}    {case.turns[-1][:52]}")
        for failure in failures:
            print(f"          - {failure}")
        # Printed for every case, pass or fail. An answer that passes the
        # automated checks can still read badly, and the checks cannot
        # see that.
        print(f"          > {reply.answer[:160].replace(chr(10), ' ')}")
        print()

    print(f"{passed}/{len(EVAL_CASES)} passed\n")

    _print_threshold_analysis(should_retrieve_scores, should_not_retrieve_scores)


def _print_threshold_analysis(
    should_retrieve: list[tuple[str, float]],
    should_not_retrieve: list[tuple[str, float]],
) -> None:
    """Print the two score groups and suggest a relevance threshold.

    Args:
        should_retrieve: (case id, top score) for questions the knowledge
            base can genuinely answer.
        should_not_retrieve: the same, for questions it cannot.
    """
    print("-" * 64)
    print("THRESHOLD ANALYSIS")
    print("-" * 64)

    if not should_retrieve or not should_not_retrieve:
        print("Not enough data in both groups to suggest a threshold.\n")
        return

    print("\nShould be answered (want these ABOVE the threshold):")
    for case_id, score in sorted(should_retrieve, key=lambda pair: -pair[1]):
        print(f"   {score:.2f}  {case_id}")

    print("\nShould be refused (want these BELOW the threshold):")
    for case_id, score in sorted(should_not_retrieve, key=lambda pair: -pair[1]):
        print(f"   {score:.2f}  {case_id}")

    worst_passer = min(score for _, score in should_retrieve)
    best_failer = max(score for _, score in should_not_retrieve)

    print(f"\nLowest score that should pass:  {worst_passer:.2f}")
    print(f"Highest score that should fail: {best_failer:.2f}")

    if worst_passer > best_failer:
        suggested = (worst_passer + best_failer) / 2
        print(f"\nClean gap. Suggested MIN_RELEVANCE_SCORE: {suggested:.2f}")
    else:
        print(
            "\nThe two groups OVERLAP, so no threshold separates them."
            "\nA question that should be refused scores higher than one that"
            "\nshould be answered. Changing the number cannot fix this - the"
            "\nsystem prompt has to catch these cases by noticing that the"
            "\nretrieved text does not actually cover what was asked."
        )

    print()


if __name__ == "__main__":
    main()
