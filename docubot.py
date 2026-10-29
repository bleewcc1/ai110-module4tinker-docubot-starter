"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import re
import glob

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Split documents into paragraph chunks and index those
        self.chunks = self.chunk_documents(self.documents)  # List of (filename, chunk)
        self.index = self.build_index(self.chunks)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in sorted(glob.glob(pattern)):  # sorted: tie order must not depend on the filesystem
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
        "from", "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
        "this", "to", "what", "where", "which", "with",
    }

    # The best chunk must match at least this fraction of the query's
    # meaningful words, or retrieval returns nothing ("no useful context").
    # A long query matching only one word is almost always an accident.
    MIN_QUERY_COVERAGE = 0.2

    def tokenize(self, text):
        """
        Lowercases text and splits it into alphanumeric words, dropping
        punctuation and common stopwords.
        """
        words = re.findall(r"[a-z0-9]+", text.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def chunk_documents(self, documents):
        """
        Splits each document into paragraphs (blocks separated by blank lines).
        A one-line heading ("## Title") or lead-in ("Headers:") is joined to
        the block after it so it stays with the text it introduces.
        Returns a list of tuples: (filename, chunk_text)
        """
        chunks = []
        for filename, text in documents:
            lead_in = ""
            for block in re.split(r"\n\s*\n", text):
                block = block.strip()
                if not block:
                    continue
                is_one_line = "\n" not in block
                if is_one_line and (block.startswith("#") or block.endswith(":")):
                    lead_in += block + "\n\n"
                    continue
                chunks.append((filename, lead_in + block))
                lead_in = ""
            if lead_in:
                chunks.append((filename, lead_in.strip()))
        return chunks

    def build_index(self, chunks):
        """
        Builds a tiny inverted index mapping lowercase words to the chunks
        they appear in, as positions in the chunks list.

        Example structure:
        {
            "token": [0, 4, 17],
            "database": [22, 23]
        }
        """
        index = {}
        for chunk_id, (_, text) in enumerate(chunks):
            for word in set(self.tokenize(text)):
                index.setdefault(word, []).append(chunk_id)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        TODO (Phase 1):
        Return a simple relevance score for how well the text matches the query.

        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score
        """
        # Distinct query words matched is the main signal, so a chunk that
        # repeats one common word doesn't beat one that covers more of the
        # query. Total occurrences only break ties (capped below 1 distinct word).
        query_words = set(self.tokenize(query))
        words = self.tokenize(text)
        distinct = len(query_words & set(words))
        occurrences = sum(1 for word in words if word in query_words)
        return distinct + occurrences / (occurrences + 1)

    def query_coverage(self, query, text):
        """
        Returns the fraction (0 to 1) of the query's distinct meaningful words
        that appear in the text. An empty query has coverage 0.
        """
        query_words = set(self.tokenize(query))
        if not query_words:
            return 0.0
        return len(query_words & set(self.tokenize(text))) / len(query_words)

    def retrieve(self, query, top_k=3):
        """
        TODO (Phase 1):
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending, or an
        empty list when there is no useful context for the query.
        """
        # Only chunks sharing at least one word with the query are candidates
        candidates = set()
        for word in set(self.tokenize(query)):
            candidates.update(self.index.get(word, []))

        results = []
        for chunk_id in sorted(candidates):
            filename, text = self.chunks[chunk_id]
            score = self.score_document(query, text)
            if score > 0:
                results.append((score, filename, text))

        results.sort(key=lambda item: item[0], reverse=True)

        # No useful context: nothing matched, or even the best chunk covers too
        # little of the query. Returning [] makes callers answer "I do not know".
        if not results:
            return []
        best_text = results[0][2]
        if self.query_coverage(query, best_text) < self.MIN_QUERY_COVERAGE:
            return []

        return [(filename, text) for _, filename, text in results[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
