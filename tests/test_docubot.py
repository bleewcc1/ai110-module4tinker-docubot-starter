"""
Tests for DocuBot's retrieval pipeline: tokenize, chunk_documents,
build_index, score_document, and retrieve.

Most tests build a DocuBot over a small temporary docs folder so they do not
depend on the contents of docs/. The last class runs against the real docs.

Run from the project root:  python -m unittest discover tests -v
"""

import atexit
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docubot import DocuBot

REAL_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")


def make_bot(files):
    """Builds a DocuBot over a temp folder holding {filename: text}."""
    folder = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    for name, text in files.items():
        with open(os.path.join(folder, name), "w", encoding="utf8") as f:
            f.write(text)
    return DocuBot(docs_folder=folder)


class TokenizeTests(unittest.TestCase):
    def setUp(self):
        self.bot = make_bot({})

    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(self.bot.tokenize("Hello, World!"), ["hello", "world"])

    def test_drops_stopwords(self):
        self.assertEqual(self.bot.tokenize("How do I refresh the token?"), ["refresh", "token"])

    def test_splits_identifiers_on_underscores(self):
        self.assertEqual(self.bot.tokenize("AUTH_SECRET_KEY"), ["auth", "secret", "key"])

    def test_empty_text(self):
        self.assertEqual(self.bot.tokenize(""), [])


class ChunkDocumentsTests(unittest.TestCase):
    def setUp(self):
        self.bot = make_bot({})

    def test_splits_on_blank_lines(self):
        chunks = self.bot.chunk_documents([("a.md", "one\n\ntwo\n\n\nthree")])
        self.assertEqual([text for _, text in chunks], ["one", "two", "three"])

    def test_keeps_filename_on_every_chunk(self):
        chunks = self.bot.chunk_documents([("a.md", "x\n\ny"), ("b.md", "z")])
        self.assertEqual([name for name, _ in chunks], ["a.md", "a.md", "b.md"])

    def test_heading_is_joined_to_next_paragraph(self):
        chunks = self.bot.chunk_documents([("a.md", "## Title\n\nBody text here.")])
        self.assertEqual(chunks, [("a.md", "## Title\n\nBody text here.")])

    def test_colon_lead_in_is_joined_to_next_block(self):
        text = "Request Body:\n\n```json\n{}\n```"
        self.assertEqual(self.bot.chunk_documents([("a.md", text)]), [("a.md", text)])

    def test_consecutive_lead_ins_are_all_kept(self):
        text = "## Login\n\nHeaders:\n\n```text\nAuthorization\n```"
        self.assertEqual(self.bot.chunk_documents([("a.md", text)]), [("a.md", text)])

    def test_multiline_paragraph_ending_in_colon_is_not_a_lead_in(self):
        chunks = self.bot.chunk_documents([("a.md", "line one\nends here:\n\nnext")])
        self.assertEqual([t for _, t in chunks], ["line one\nends here:", "next"])

    def test_trailing_heading_is_kept(self):
        chunks = self.bot.chunk_documents([("a.md", "Body.\n\n## Orphan")])
        self.assertEqual([text for _, text in chunks], ["Body.", "## Orphan"])

    def test_multiline_block_stays_together(self):
        text = "- item one\n- item two\n- item three"
        chunks = self.bot.chunk_documents([("a.md", text)])
        self.assertEqual(chunks, [("a.md", text)])

    def test_empty_document_yields_no_chunks(self):
        self.assertEqual(self.bot.chunk_documents([("a.md", "  \n\n  \n")]), [])

    def test_documents_load_in_sorted_order(self):
        bot = make_bot({"b.md": "x", "c.md": "y", "a.md": "z"})
        self.assertEqual([name for name, _ in bot.documents], ["a.md", "b.md", "c.md"])

    def test_loaded_docs_are_chunked_on_init(self):
        bot = make_bot({"a.md": "alpha\n\nbeta"})
        self.assertEqual(len(bot.documents), 1)
        self.assertEqual(len(bot.chunks), 2)


class BuildIndexTests(unittest.TestCase):
    def setUp(self):
        self.bot = make_bot({})
        self.chunks = [
            ("a.md", "database connection string"),
            ("b.md", "token token token"),
            ("c.md", "database token"),
        ]
        self.index = self.bot.build_index(self.chunks)

    def test_maps_word_to_chunk_positions(self):
        self.assertEqual(self.index["database"], [0, 2])

    def test_chunk_listed_once_per_word_even_if_repeated(self):
        self.assertEqual(self.index["token"], [1, 2])

    def test_lowercases_words(self):
        index = self.bot.build_index([("a.md", "Token TOKEN")])
        self.assertEqual(index, {"token": [0]})

    def test_stopwords_are_not_indexed(self):
        index = self.bot.build_index([("a.md", "the and of")])
        self.assertEqual(index, {})

    def test_unknown_word_is_absent(self):
        self.assertNotIn("payment", self.index)

    def test_empty_chunks(self):
        self.assertEqual(self.bot.build_index([]), {})


class ScoreDocumentTests(unittest.TestCase):
    def setUp(self):
        self.bot = make_bot({})

    def test_no_overlap_scores_zero(self):
        self.assertEqual(self.bot.score_document("payment refunds", "token and database"), 0)

    def test_stopword_only_query_scores_zero(self):
        self.assertEqual(self.bot.score_document("how do I", "how do I refresh"), 0)

    def test_matching_scores_above_zero(self):
        self.assertGreater(self.bot.score_document("token", "a token here"), 0)

    def test_more_distinct_query_words_scores_higher(self):
        one = self.bot.score_document("refresh token", "token")
        two = self.bot.score_document("refresh token", "refresh the token")
        self.assertGreater(two, one)

    def test_repeating_one_word_does_not_beat_covering_two(self):
        repeated = self.bot.score_document("refresh token", "token token token token token")
        covered = self.bot.score_document("refresh token", "refresh token")
        self.assertGreater(covered, repeated)

    def test_more_occurrences_breaks_a_tie(self):
        few = self.bot.score_document("token", "token")
        many = self.bot.score_document("token", "token token token")
        self.assertGreater(many, few)

    def test_case_insensitive(self):
        self.assertEqual(
            self.bot.score_document("TOKEN", "token"),
            self.bot.score_document("token", "TOKEN"),
        )


class RetrieveTests(unittest.TestCase):
    def setUp(self):
        self.bot = make_bot({
            "auth.md": "# Auth\n\nRefresh the token before it expires.\n\nTokens are signed with a secret.",
            "db.md": "# Database\n\nThe users table stores email and password hash.\n\nUse sqlite locally.",
            "setup.md": "# Setup\n\nInstall dependencies with pip.",
        })

    def test_returns_filename_text_tuples(self):
        results = self.bot.retrieve("refresh token")
        self.assertTrue(results)
        for item in results:
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], str)

    def test_best_chunk_comes_first(self):
        filename, text = self.bot.retrieve("refresh token")[0]
        self.assertEqual(filename, "auth.md")
        self.assertIn("Refresh the token", text)

    def test_results_are_sorted_by_score_descending(self):
        query = "users table email"
        scores = [self.bot.score_document(query, text) for _, text in self.bot.retrieve(query)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_different_queries_return_different_results(self):
        auth = self.bot.retrieve("refresh token")[0]
        db = self.bot.retrieve("users table password")[0]
        pip = self.bot.retrieve("install pip dependencies")[0]
        self.assertEqual(auth[0], "auth.md")
        self.assertEqual(db[0], "db.md")
        self.assertEqual(pip[0], "setup.md")

    def test_top_k_limits_results(self):
        self.assertEqual(len(self.bot.retrieve("the token users pip", top_k=1)), 1)
        self.assertLessEqual(len(self.bot.retrieve("token users pip sqlite", top_k=2)), 2)

    def test_no_match_returns_empty_list(self):
        self.assertEqual(self.bot.retrieve("payment refunds"), [])

    def test_empty_query_returns_empty_list(self):
        self.assertEqual(self.bot.retrieve(""), [])

    def test_retrieval_only_answer_says_unknown_when_nothing_matches(self):
        self.assertEqual(
            self.bot.answer_retrieval_only("payment refunds"),
            "I do not know based on these docs.",
        )

    def test_retrieval_only_answer_includes_filename_label(self):
        self.assertIn("[auth.md]", self.bot.answer_retrieval_only("refresh token"))


class QueryCoverageTests(unittest.TestCase):
    def setUp(self):
        self.bot = make_bot({})

    def test_full_match_is_one(self):
        self.assertEqual(self.bot.query_coverage("refresh token", "refresh the token"), 1.0)

    def test_partial_match_is_fraction_of_query_words(self):
        self.assertEqual(self.bot.query_coverage("refresh token expiry now", "refresh token"), 0.5)

    def test_no_match_is_zero(self):
        self.assertEqual(self.bot.query_coverage("payment", "token"), 0.0)

    def test_empty_or_stopword_only_query_is_zero(self):
        self.assertEqual(self.bot.query_coverage("", "token"), 0.0)
        self.assertEqual(self.bot.query_coverage("how do I", "how do I"), 0.0)

    def test_repeats_in_text_do_not_inflate_coverage(self):
        self.assertEqual(self.bot.query_coverage("token refresh", "token token token"), 0.5)


class NoUsefulContextTests(unittest.TestCase):
    def setUp(self):
        self.bot = make_bot({
            "a.md": "The token is signed.\n\nPayment is out of scope.",
            "b.md": "Install dependencies with pip.",
        })

    def test_weak_match_on_a_long_query_returns_nothing(self):
        # Only "payment" (1 of 7 meaningful words) matches anywhere: 14% < 20%
        query = "Is there any mention of payment processing in these docs?"
        self.assertEqual(self.bot.retrieve(query), [])
        self.assertEqual(
            self.bot.answer_retrieval_only(query), "I do not know based on these docs."
        )

    def test_same_single_word_match_is_enough_for_a_short_query(self):
        # "payment" is 1 of 2 meaningful words: 50% >= 20%
        results = self.bot.retrieve("payment refunds")
        self.assertEqual([f for f, _ in results], ["a.md"])

    def test_query_exactly_at_threshold_is_kept(self):
        # 1 of 5 meaningful words = 20%, which is allowed
        results = self.bot.retrieve("pip alpha beta gamma delta")
        self.assertEqual([f for f, _ in results], ["b.md"])

    def test_query_just_below_threshold_is_dropped(self):
        # 1 of 6 meaningful words is about 17%
        self.assertEqual(self.bot.retrieve("pip alpha beta gamma delta epsilon"), [])

    def test_rag_does_not_call_llm_without_context(self):
        class FailingLLM:
            def answer_from_snippets(self, query, snippets):
                raise AssertionError("LLM should not be called without context")

        bot = DocuBot(docs_folder=self.bot.docs_folder, llm_client=FailingLLM())
        query = "Is there any mention of payment processing in these docs?"
        self.assertEqual(bot.answer_rag(query), "I do not know based on these docs.")


class RealDocsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = DocuBot(docs_folder=REAL_DOCS)

    def test_loads_all_docs_and_splits_them_into_chunks(self):
        names = {name for name, _ in self.bot.documents}
        self.assertTrue({"AUTH.md", "API_REFERENCE.md", "DATABASE.md", "SETUP.md"} <= names)
        self.assertGreater(len(self.bot.chunks), len(self.bot.documents))

    def test_users_endpoint_query_finds_api_reference(self):
        filename, text = self.bot.retrieve("Which endpoint lists all users?")[0]
        self.assertEqual(filename, "API_REFERENCE.md")
        self.assertIn("/api/users", text)

    def test_auth_secret_query_finds_auth_doc(self):
        files = [f for f, _ in self.bot.retrieve("What does AUTH_SECRET_KEY sign?")]
        self.assertIn("AUTH.md", files)

    def test_unrelated_query_returns_nothing(self):
        self.assertEqual(self.bot.retrieve("kubernetes helm chart"), [])

    def test_payment_processing_query_returns_nothing(self):
        query = "Is there any mention of payment processing in these docs?"
        self.assertEqual(self.bot.retrieve(query), [])

    def test_answerable_queries_still_return_results(self):
        for query in [
            "Where is the auth token generated?",
            "How do I connect to the database?",
            "Which fields are stored in the users table?",
            "How does a client refresh an access token?",
        ]:
            with self.subTest(query=query):
                self.assertTrue(self.bot.retrieve(query))


if __name__ == "__main__":
    unittest.main()
