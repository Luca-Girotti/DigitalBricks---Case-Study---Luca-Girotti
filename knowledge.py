"""
The knowledge base: loading Markdown documents, splitting them into
chunks, and searching them.

This file has three jobs, in order:

1. Read every .md file in knowledge_base/ off disk.
2. Split each one into chunks, one per "## " heading.
3. Turn each chunk into a vector, and find the chunks most similar to a
   question.

Why split on headings rather than by character count?

The common approach is to cut the text every N characters with some
overlap. It is easy to write, but it cuts sentences in half and produces
chunks that begin mid-thought. It also gives you nothing precise to cite
with: you can name the file, but not the part of it.

Our documents are already written with meaningful headings, so a heading
is a natural boundary - the author has already told us where one idea
ends and the next begins. Each chunk is a complete, self-contained
section, and it arrives with its own title, which becomes the citation
for free: "After-Hours Callout Policy, section 'Who approves an
after-hours callout'".

The trade-off is real and worth stating: a very long section becomes one
large chunk, which dilutes its embedding and can make it harder to match.
We accept that because we control these documents and keep sections
short. Against a large uncontrolled corpus this strategy would need a
maximum chunk size with a fallback split.
"""

from dataclasses import dataclass, field
from pathlib import Path

import config


@dataclass
class Chunk:
    """One searchable section of one document.

    Attributes:
        document: The document's human-readable title, taken from its
            top-level "# " heading. Used in citations.
        section: The "## " heading this chunk sits under. Also used in
            citations, so the assistant can point at the exact part.
        text: The section's body text, preceded by the document title
            and the section heading. This is what gets embedded and what
            the model reads.
        embedding: The vector representation, filled in later by
            build_index(). Empty until then.
    """

    document: str
    section: str
    text: str
    embedding: list[float] = field(default_factory=list)

    def citation(self) -> str:
        """Return a human-readable source reference for this chunk.

        Returns:
            A string like "After-Hours Callout Policy > Callout rates",
            which the model is instructed to quote when it uses this
            chunk in an answer.
        """
        return f"{self.document} > {self.section}"


def split_into_chunks(markdown_text: str, fallback_title: str) -> list[Chunk]:
    """Split one Markdown document into one chunk per "## " heading.

    Walks the document line by line. A line starting with "# " sets the
    document title. A line starting with "## " closes the current chunk
    and opens a new one. Everything else is body text appended to
    whichever chunk is currently open.

    Any text appearing before the first "## " heading (the document's
    front matter - owner, review date, scope) is kept as a chunk called
    "Overview" rather than thrown away, because it sometimes answers
    questions about which sites a policy applies to.

    Args:
        markdown_text: The full text of the document.
        fallback_title: Title to use if the document has no "# " heading,
            normally derived from the filename.

    Returns:
        A list of Chunk objects with empty embeddings. Chunks whose body
        is blank are skipped.
    """
    chunks: list[Chunk] = []

    document_title = fallback_title
    current_section = "Overview"
    current_lines: list[str] = []

    def close_current_chunk() -> None:
        """Save whatever has been collected so far, if it is not blank.

        Defined inside split_into_chunks because it needs access to the
        loop's running state, and is called from two places below - when
        a new heading arrives, and once more at the end of the file.
        """
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append(
                Chunk(
                    document=document_title,
                    section=current_section,
                    # Both the document title and the section heading are
                    # included in the embedded text, not just the body.
                    #
                    # The section heading alone was not enough. The
                    # evaluation showed "What is the SLA for a P2 ticket?"
                    # retrieving the Client complaints section, and
                    # "P2 SLA" retrieving P4 - Low. The P2 chunk never
                    # contained the words "Service Level Agreement" -
                    # those lived only in the document title - so the four
                    # priority chunks looked almost identical to the
                    # embedding model. Prepending the title gives every
                    # chunk back the context it lost when it was cut out
                    # of its document.
                    text=f"# {document_title}\n## {current_section}\n\n{body}",
                )
            )

    for line in markdown_text.splitlines():
        if line.startswith("# "):
            # The document's own title. Not a chunk boundary.
            document_title = line[2:].strip()
        elif line.startswith("## "):
            # A new section begins: close the previous one first.
            close_current_chunk()
            current_section = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    # The final section has no heading after it to trigger the close.
    close_current_chunk()

    return chunks


def load_chunks(directory: Path | None = None) -> list[Chunk]:
    """Read every Markdown file in the knowledge base and chunk them all.

    Args:
        directory: Folder to read from. Defaults to the path in
            config.KNOWLEDGE_BASE_DIR. Overridable mainly so tests can
            point at a different folder.

    Returns:
        A flat list of every chunk from every document, sorted by
        filename so the ordering is stable between runs.

    Raises:
        FileNotFoundError: if the directory does not exist, with a
            message naming the path that was tried.
    """
    directory = directory or config.KNOWLEDGE_BASE_DIR

    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {directory}")

    all_chunks: list[Chunk] = []

    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # "after-hours-callouts.md" -> "After Hours Callouts", used only
        # if the file forgot its "# " heading.
        fallback_title = path.stem.replace("-", " ").title()
        all_chunks.extend(split_into_chunks(text, fallback_title))

    return all_chunks


# --------------------------------------------------------------------------
# Embedding and search
#
# Everything below this line needs an Azure connection. Everything above
# it - loading and chunking - runs offline.
# --------------------------------------------------------------------------


@dataclass
class SearchResult:
    """What one search returned.

    Attributes:
        matches: The chunks that scored at or above the relevance
            threshold, best first, each paired with its score. Empty when
            nothing was good enough.
        top_score: The best score seen, whether or not it cleared the
            threshold. Kept separately because evaluate.py needs to see
            the near-misses in order to tune the threshold - if we only
            reported what passed, we could never tell how close a failed
            search came.
    """

    matches: list[tuple[Chunk, float]]
    top_score: float


def similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors.

    Cosine similarity is normally the dot product divided by both
    vectors' lengths. Azure's text-embedding-3 models return vectors that
    are already unit length, so that division is by 1 x 1 and disappears,
    leaving just the dot product.

    Args:
        a: First vector.
        b: Second vector, the same length as the first.

    Returns:
        A score, in practice between roughly 0.0 (unrelated) and 1.0
        (the same meaning).
    """
    return sum(x * y for x, y in zip(a, b))


def embed(texts: list[str]) -> list[list[float]]:
    """Turn a list of strings into a list of vectors.

    Sent as a single batched request rather than one call per string.
    All 37 chunks go to Azure in one round trip, which is both faster and
    cheaper than 37 separate calls.

    Args:
        texts: The strings to embed.

    Returns:
        One vector per input string, in the same order.
    """
    response = config.get_client().embeddings.create(
        model=config.EMBEDDING_DEPLOYMENT,
        input=texts,
    )
    return [item.embedding for item in response.data]


# The built index, held here after the first search so that we embed the
# knowledge base once per process rather than once per question.
_index: list[Chunk] | None = None


def get_index() -> list[Chunk]:
    """Return the chunks with their embeddings filled in.

    Builds the index on first call - reading the documents, chunking
    them, and embedding every chunk in one batched request - then reuses
    it for the rest of the process.

    Rebuilding per question would mean an Azure call and a few seconds of
    delay every single time, for a knowledge base that does not change
    while the program is running.

    Returns:
        The list of Chunk objects, each with a populated embedding.
    """
    global _index

    if _index is None:
        chunks = load_chunks()
        vectors = embed([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
        _index = chunks

    return _index


def search(
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
) -> SearchResult:
    """Find the chunks most relevant to a question.

    Embeds the query, scores it against every chunk, keeps the best few,
    and discards anything below the relevance threshold.

    That last step is the one that matters. Similarity search always
    returns a ranking, even when nothing in the corpus is relevant - ask
    about parking and it will still hand back the three least unrelated
    chunks. Passing those to the model invites it to build a plausible
    answer out of unrelated text. The threshold throws them away first,
    so the model sees an empty result and can say it does not know.

    Args:
        query: The user's question, or whatever the model chose to search
            for on their behalf.
        top_k: How many chunks to keep. Defaults to config.SEARCH_TOP_K.
        min_score: The relevance floor. Defaults to
            config.MIN_RELEVANCE_SCORE.

    Returns:
        A SearchResult. Its matches list is empty when nothing cleared
        the threshold, but top_score still reports how close the best
        candidate came.
    """
    top_k = top_k if top_k is not None else config.SEARCH_TOP_K
    min_score = min_score if min_score is not None else config.MIN_RELEVANCE_SCORE

    chunks = get_index()

    # embed() takes and returns lists, so ask for one and unwrap it.
    query_vector = embed([query])[0]

    scored = [(chunk, similarity(query_vector, chunk.embedding)) for chunk in chunks]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    # Recorded before filtering, so a failed search can still report how
    # near it got.
    top_score = scored[0][1] if scored else 0.0

    matches = [(chunk, score) for chunk, score in scored[:top_k] if score >= min_score]

    return SearchResult(matches=matches, top_score=top_score)


if __name__ == "__main__":
    # Run "python knowledge.py" to see exactly how the documents were
    # split. Useful when tuning the chunking strategy, and a quick way to
    # show the retrieval layer during a demo without needing Azure.
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {config.KNOWLEDGE_BASE_DIR}\n")
    for chunk in chunks:
        preview = chunk.text.split("\n\n", 1)[-1][:70].replace("\n", " ")
        print(f"  {len(chunk.text):>5} chars | {chunk.citation()}")
        print(f"         | {preview}...")
