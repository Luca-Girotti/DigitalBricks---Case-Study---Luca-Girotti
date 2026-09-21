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

Second, it produces the numbers needed to set MIN_RELEVANCE_SCORE, by
printing the best retrieval score for three groups of question:

- answerable      - the knowledge base covers it
- near-domain     - sounds like it belongs, but the knowledge base does
                    not cover it (parking, holiday, a site we lack)
- off-topic       - nothing to do with facilities at all (the weather)

If the answerable and near-domain groups separate cleanly, the threshold
goes between them and does all the refusing on its own. On this corpus
they overlap - a question about a site Northwind does not have scores
higher than some genuine questions - so the system prompt has to catch
near-domain questions, and the threshold's job shrinks to keeping
off-topic noise away from the model without ever blocking a real answer.
The summary says which situation applies and suggests a number for it.
"""

from agent import AgentReply, Assistant
from eval_questions import EVAL_CASES, OFF_TOPIC_PROBES, EvalCase
import config
import knowledge


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
    """Run one case end to end.

    A fresh Assistant per case, so cases cannot contaminate each other.
    Multi-turn cases run every turn in order.

    The answer checked is the reply to the last turn. The tools and
    sources, though, are gathered from the whole conversation. The system
    prompt defines grounded as "text a tool returned in this
    conversation", so a follow-up answered from a section retrieved one
    turn earlier is grounded, not a guess. An earlier version counted the
    last turn only, and failed M1 whenever the model - correctly - reused
    what it had already retrieved instead of searching again.

    Args:
        case: The case to run.

    Returns:
        An AgentReply combining the final answer with the tools, sources
        and best score from every turn.
    """
    assistant = Assistant()
    replies = [assistant.ask(turn) for turn in case.turns]

    return AgentReply(
        answer=replies[-1].answer,
        tools_called=[tool for reply in replies for tool in reply.tools_called],
        sources=[source for reply in replies for source in reply.sources],
        top_score=max(reply.top_score for reply in replies),
    )


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
    """Print the three score groups and suggest a relevance threshold.

    Args:
        should_retrieve: (case id, top score) for questions the knowledge
            base can genuinely answer.
        should_not_retrieve: the same, for near-domain questions it
            cannot.
    """
    print("-" * 64)
    print("THRESHOLD ANALYSIS")
    print("-" * 64)

    if not should_retrieve or not should_not_retrieve:
        print("Not enough data in both groups to suggest a threshold.\n")
        return

    print("\nAnswerable (must stay ABOVE the threshold):")
    for case_id, score in sorted(should_retrieve, key=lambda pair: -pair[1]):
        print(f"   {score:.2f}  {case_id}")

    print("\nNear-domain, should be refused:")
    for case_id, score in sorted(should_not_retrieve, key=lambda pair: -pair[1]):
        print(f"   {score:.2f}  {case_id}")

    # Off-topic questions are scored straight against the index rather
    # than run through the agent, because the agent never searches for
    # them - it declines without calling a tool at all. What we want to
    # know is what the threshold WOULD see if a search were made, since
    # that is the situation a backstop exists for.
    off_topic = [(query, knowledge.search(query).top_score) for query in OFF_TOPIC_PROBES]
    print("\nOff-topic, scored directly against the index:")
    for query, score in sorted(off_topic, key=lambda pair: -pair[1]):
        print(f"   {score:.2f}  {query}")

    worst_passer = min(score for _, score in should_retrieve)
    best_failer = max(score for _, score in should_not_retrieve)
    noisiest_off_topic = max(score for _, score in off_topic)

    print(f"\nLowest answerable:        {worst_passer:.2f}")
    print(f"Highest near-domain:      {best_failer:.2f}")
    print(f"Highest off-topic:        {noisiest_off_topic:.2f}")

    if worst_passer > best_failer:
        suggested = (worst_passer + best_failer) / 2
        print(
            "\nAnswerable and near-domain separate cleanly, so the threshold"
            "\ncan do all the refusing by itself."
            f"\nSuggested MIN_RELEVANCE_SCORE: {suggested:.2f}"
        )
    elif worst_passer > noisiest_off_topic:
        suggested = (worst_passer + noisiest_off_topic) / 2
        print(
            "\nAnswerable and near-domain OVERLAP, so no threshold can"
            "\nseparate them - the system prompt has to refuse those, by"
            "\nnoticing the retrieved text does not cover what was asked."
            "\n"
            "\nThe threshold's job is therefore to keep off-topic noise away"
            "\nfrom the model without ever blocking a real answer: the"
            "\nmidpoint between the highest off-topic and lowest answerable."
            f"\nSuggested MIN_RELEVANCE_SCORE: {suggested:.2f}"
        )
    else:
        print(
            "\nEven off-topic questions score as high as answerable ones."
            "\nNo threshold is safe here. That points at the documents or"
            "\nthe chunking, not at this number."
        )

    print()


if __name__ == "__main__":
    main()
