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
        text: The section's body text with its heading included. This is
            what gets embedded and what the model reads.
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
                    # The heading is deliberately included in the text.
                    # "Who approves an after-hours callout" is a strong
                    # signal for the embedding, and losing it would make
                    # the chunk noticeably harder to match.
                    text=f"## {current_section}\n\n{body}",
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
