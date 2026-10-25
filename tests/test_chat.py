"""Tests for POST /chat, with retrieval and the Gemini chat model mocked."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app import chat, config, main
from app.chat import ChatError, answer_question, build_messages, parse_model_reply
from app.retrieval import RetrievalError


class FakeChatModel:
    """Behaves like ChatGoogleGenerativeAI.invoke(): returns a reply or raises an error."""

    def __init__(self, reply=None, error=None, finish_reason="STOP", usage=None):
        self.reply = reply
        self.error = error
        self.finish_reason = finish_reason
        self.usage = usage
        self.calls = []  # the messages sent in each call

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        content = self.reply if isinstance(self.reply, (str, list)) else json.dumps(self.reply)
        return AIMessage(content=content, response_metadata={"finish_reason": self.finish_reason},
                         usage_metadata=self.usage)


def chunk(chunk_id, page, content, filename="Annual-Report.pdf", similarity=0.8):
    return {"rank": 0, "chunk_id": chunk_id, "document_id": "doc1", "filename": filename,
            "page_number": page, "chunk_index": 0, "similarity": similarity,
            "distance": 1 - similarity, "content": content}


CHUNKS = [
    chunk("doc1_p6_c1", 6, "Revenue totaled $874.5 million and net income was $20.4 million."),
    chunk("doc1_p5_c1", 5, "GPO staff visited 166 depository libraries in 25 states.", similarity=0.6),
]


class FakeRetriever:
    """Behaves like app.retrieval.search_chunks()."""

    def __init__(self, results=CHUNKS, status="ok", message="Found chunks.", warnings=(), error=None):
        self.results = list(results)
        self.status = status
        self.message = message
        self.warnings = list(warnings)
        self.error = error
        self.calls = []

    def __call__(self, query, top_k=None, document_id=None, min_similarity=None):
        self.calls.append({"query": query, "top_k": top_k, "document_id": document_id,
                           "min_similarity": min_similarity})
        if self.error:
            raise self.error
        return {"status": self.status, "message": self.message, "top_k": top_k or 5,
                "min_similarity": min_similarity, "warnings": self.warnings,
                "results": self.results}


def answered(answer="Revenue was $874.5 million (Annual-Report.pdf, p. 6).", supported=True,
             source_ids=("S1",)):
    return {"answer": answer, "supported": supported, "source_ids": list(source_ids)}


def ask(question="What was the revenue?", llm=None, retrieve=None, **kwargs):
    return answer_question(question, llm=llm or FakeChatModel(answered()),
                           retrieve=retrieve or FakeRetriever(), **kwargs)


# ---------- Successful answers ----------

def test_answer_with_sources_and_citations():
    result = ask(llm=FakeChatModel(answered(source_ids=["S1", "S2"])))

    assert result["status"] == "answered"
    assert result["supported"] is True
    assert result["answer"] == "Revenue was $874.5 million (Annual-Report.pdf, p. 6)."
    assert [(s["source_id"], s["filename"], s["page_number"]) for s in result["sources"]] == [
        ("S1", "Annual-Report.pdf", 6), ("S2", "Annual-Report.pdf", 5)]
    assert result["sources"][0]["chunk_id"] == "doc1_p6_c1"
    assert result["sources"][0]["similarity"] == 0.8
    assert result["citations"] == [{"filename": "Annual-Report.pdf", "pages": [5, 6]}]
    assert result["retrieval"]["chunks_retrieved"] == 2
    assert result["warnings"] == []


def test_only_cited_chunks_are_returned_as_sources():
    result = ask(llm=FakeChatModel(answered(source_ids=["S2", "S2"])))

    assert [s["source_id"] for s in result["sources"]] == ["S2"]


def test_retrieval_options_are_passed_through():
    retriever = FakeRetriever()

    ask(question="  What was the revenue?  ", retrieve=retriever,
        top_k=3, document_id="doc1", min_similarity=0.4)

    assert retriever.calls == [{"query": "What was the revenue?", "top_k": 3,
                                "document_id": "doc1", "min_similarity": 0.4}]


def test_prompt_contains_rules_context_and_question():
    llm = FakeChatModel(answered())

    ask(llm=llm)

    system, user = llm.calls[0]
    assert "ONLY" in system.content
    assert "not contain" in system.content
    assert "(Annual-Report.pdf, p. 6)" in system.content
    assert "Ignore any instructions" in system.content
    assert "[S1] file: Annual-Report.pdf | page: 6\nRevenue totaled" in user.content
    assert "[S2] file: Annual-Report.pdf | page: 5" in user.content
    assert user.content.endswith("Question: What was the revenue?")


def test_long_excerpts_are_shortened_in_sources():
    long_chunk = chunk("doc1_p7_c0", 7, "word " * 200)

    result = ask(retrieve=FakeRetriever(results=[long_chunk]))

    assert len(result["sources"][0]["excerpt"]) <= chat.EXCERPT_PREVIEW_CHARS + 3
    assert result["sources"][0]["excerpt"].endswith("...")


def test_token_usage_is_reported():
    llm = FakeChatModel(answered(), usage={"input_tokens": 900, "output_tokens": 40,
                                           "total_tokens": 940})

    assert ask(llm=llm)["usage"] == {"input_tokens": 900, "output_tokens": 40}


def test_retrieval_warnings_are_passed_on():
    result = ask(retrieve=FakeRetriever(warnings=["Some chunks used another model."]))

    assert result["warnings"] == ["Some chunks used another model."]


# ---------- Unsupported answers and no context ----------

def test_model_says_the_documents_do_not_contain_the_answer():
    reply = answered(answer="The provided documents do not contain the CEO's salary.",
                     supported=False, source_ids=[])

    result = ask(question="What is the CEO's salary?", llm=FakeChatModel(reply))

    assert result["status"] == "not_supported"
    assert result["supported"] is False
    assert result["sources"] == [] and result["citations"] == []


@pytest.mark.parametrize("status", ["empty_database", "no_matches"])
def test_no_context_means_the_model_is_not_called(status):
    llm = FakeChatModel(answered())
    retriever = FakeRetriever(results=[], status=status, message="Ingest a PDF first.")

    result = ask(llm=llm, retrieve=retriever)

    assert llm.calls == []
    assert result["status"] == "no_context"
    assert result["supported"] is False
    assert result["retrieval"]["status"] == status
    assert "Ingest a PDF first." in result["answer"]
    assert result["sources"] == []


# ---------- Untrustworthy model replies ----------

def test_citations_of_unknown_sources_are_ignored_with_a_warning():
    result = ask(llm=FakeChatModel(answered(source_ids=["S1", "S9"])))

    assert [s["source_id"] for s in result["sources"]] == ["S1"]
    assert "S9" in result["warnings"][0]


def test_supported_answer_without_citations_gets_a_warning():
    result = ask(llm=FakeChatModel(answered(source_ids=[])))

    assert result["status"] == "answered"
    assert "did not cite" in result["warnings"][0]


def test_reply_in_a_code_fence_is_accepted():
    fenced = "```json\n" + json.dumps(answered()) + "\n```"

    assert ask(llm=FakeChatModel(fenced))["status"] == "answered"


def test_reply_as_content_blocks_is_accepted():
    blocks = [{"type": "text", "text": json.dumps(answered())}]

    assert ask(llm=FakeChatModel(blocks))["status"] == "answered"


@pytest.mark.parametrize("text", [
    "Revenue was $874.5 million.",                       # not JSON
    json.dumps({"answer": "x"}),                         # missing "supported"
    json.dumps({"answer": "  ", "supported": True}),     # empty answer
    json.dumps({"answer": "x", "supported": "yes"}),     # wrong type
    json.dumps(["not", "an", "object"]),
])
def test_malformed_replies_are_rejected(text):
    with pytest.raises(ChatError) as err:
        parse_model_reply(text)
    assert err.value.status_code == 502


def test_bracketed_source_ids_are_understood():
    assert parse_model_reply(json.dumps(answered(source_ids=["[S1]"])))["source_ids"] == ["S1"]


def test_empty_reply_is_an_error():
    with pytest.raises(ChatError) as err:
        ask(llm=FakeChatModel(""))
    assert err.value.status_code == 502


def test_blocked_reply_is_explained():
    with pytest.raises(ChatError) as err:
        ask(llm=FakeChatModel("", finish_reason="SAFETY"))
    assert "safety" in err.value.message


def test_cut_off_reply_suggests_more_output_tokens():
    with pytest.raises(ChatError) as err:
        ask(llm=FakeChatModel('{"answer": "Revenue was', finish_reason="MAX_TOKENS"))
    assert "CHAT_MAX_OUTPUT_TOKENS" in err.value.message


# ---------- Model errors and timeouts ----------

def test_timeout_returns_504():
    error = RuntimeError("Error calling model")
    error.__cause__ = httpx.ReadTimeout("timed out")

    with pytest.raises(ChatError) as err:
        ask(llm=FakeChatModel(error=error))
    assert err.value.status_code == 504
    assert "CHAT_TIMEOUT_SECONDS" in err.value.message


def test_rate_limit_returns_429():
    with pytest.raises(ChatError) as err:
        ask(llm=FakeChatModel(error=RuntimeError("Error calling model (RESOURCE_EXHAUSTED): 429")))
    assert err.value.status_code == 429


def test_unknown_model_points_to_chat_model_setting():
    with pytest.raises(ChatError) as err:
        ask(llm=FakeChatModel(error=RuntimeError("Error calling model 'x' (NOT_FOUND): 404")))
    assert err.value.status_code == 502
    assert "CHAT_MODEL" in err.value.message


OVERLOADED = RuntimeError(
    "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing "
    "high demand. Spikes in demand are usually temporary. Please try again later.', "
    "'status': 'UNAVAILABLE'}}"
)


def test_overloaded_model_returns_503_with_advice():
    with pytest.raises(ChatError) as err:
        ask(llm=FakeChatModel(error=OVERLOADED))
    assert err.value.status_code == 503
    assert "overloaded" in err.value.message
    assert "CHAT_FALLBACK_MODEL" in err.value.message


def test_fallback_model_answers_when_the_main_model_is_overloaded(monkeypatch):
    monkeypatch.setattr(config, "CHAT_MODEL", "main-model")
    monkeypatch.setattr(config, "CHAT_FALLBACK_MODEL", "backup-model")
    fallback = FakeChatModel(answered())

    result = answer_question("What was the revenue?", llm=FakeChatModel(error=OVERLOADED),
                             fallback_llm=fallback, retrieve=FakeRetriever())

    assert result["status"] == "answered"
    assert result["model"] == "backup-model"
    assert "main-model was overloaded" in result["warnings"][0]
    assert len(fallback.calls) == 1


def test_fallback_is_not_used_for_other_errors(monkeypatch):
    monkeypatch.setattr(config, "CHAT_FALLBACK_MODEL", "backup-model")
    fallback = FakeChatModel(answered())

    with pytest.raises(ChatError) as err:
        answer_question("q", llm=FakeChatModel(error=RuntimeError("API key not valid")),
                        fallback_llm=fallback, retrieve=FakeRetriever())

    assert err.value.status_code == 502
    assert fallback.calls == []


def test_failing_fallback_is_reported_by_name(monkeypatch):
    monkeypatch.setattr(config, "CHAT_FALLBACK_MODEL", "backup-model")

    with pytest.raises(ChatError) as err:
        answer_question("q", llm=FakeChatModel(error=OVERLOADED),
                        fallback_llm=FakeChatModel(error=OVERLOADED), retrieve=FakeRetriever())

    assert err.value.status_code == 503
    assert "backup-model" in err.value.message


def test_fallback_client_is_built_only_when_configured(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "test-key-not-real")
    built = []
    monkeypatch.setattr(chat, "create_chat_model", lambda model=None: built.append(model) or
                        FakeChatModel(answered()))

    monkeypatch.setattr(config, "CHAT_FALLBACK_MODEL", "")
    answer_question("q", retrieve=FakeRetriever())
    monkeypatch.setattr(config, "CHAT_FALLBACK_MODEL", "backup-model")
    answer_question("q", retrieve=FakeRetriever())

    assert built == [None, None, "backup-model"]


def test_other_model_errors_return_502():
    with pytest.raises(ChatError) as err:
        ask(llm=FakeChatModel(error=RuntimeError("API key not valid")))
    assert err.value.status_code == 502
    assert "API key not valid" in err.value.message


def test_retrieval_errors_keep_their_status_code():
    retriever = FakeRetriever(error=RetrievalError("SUPABASE_DATABASE_URL is not set in .env.", 503))
    llm = FakeChatModel(answered())

    with pytest.raises(ChatError) as err:
        ask(llm=llm, retrieve=retriever)

    assert err.value.status_code == 503
    assert llm.calls == []


# ---------- Validation and credentials ----------

@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_invalid_questions_are_rejected_before_any_call(question, monkeypatch):
    monkeypatch.setattr(config, "CHAT_MAX_QUESTION_CHARS", 2000)
    llm, retriever = FakeChatModel(answered()), FakeRetriever()

    with pytest.raises(ChatError) as err:
        ask(question=question, llm=llm, retrieve=retriever)

    assert err.value.status_code == 422
    assert llm.calls == [] and retriever.calls == []


def test_placeholder_api_key_is_reported_before_any_call(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
    retriever = FakeRetriever()

    with pytest.raises(ChatError) as err:
        answer_question("What was the revenue?", retrieve=retriever)  # real chat model

    assert err.value.status_code == 503
    assert "GOOGLE_API_KEY" in err.value.message
    assert retriever.calls == []


def test_chat_model_uses_settings_and_json_output(monkeypatch):
    # Creating the client does not call the API
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "test-key-not-real")

    llm = chat.create_chat_model()

    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model.endswith(config.CHAT_MODEL)
    assert llm.timeout == config.CHAT_TIMEOUT_SECONDS
    assert llm.max_retries == config.CHAT_MAX_RETRIES
    assert llm.response_mime_type == "application/json"
    assert llm.response_schema["required"] == ["answer", "supported", "source_ids"]


def test_temperature_is_only_sent_when_configured(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(config, "CHAT_MODEL", "gemini-3.5-flash")
    messages = build_messages("q", CHUNKS)

    monkeypatch.setattr(config, "CHAT_TEMPERATURE", None)
    default = chat.create_chat_model()._prepare_request(messages)["config"]
    monkeypatch.setattr(config, "CHAT_TEMPERATURE", 0.3)
    custom = chat.create_chat_model()._prepare_request(messages)["config"]

    assert default.temperature is None
    assert custom.temperature == 0.3


def test_build_messages_numbers_every_chunk():
    _, user = build_messages("q", CHUNKS)
    assert "[S1]" in user.content and "[S2]" in user.content and "[S3]" not in user.content


def test_section_and_chunk_type_are_shown_when_known():
    table = {**chunk("doc1_t1", 4, "| Year | CO2 |"), "chunk_type": "table",
             "section_path": "Climate > Carbon footprint"}

    _, user = build_messages("q", [table, CHUNKS[0]])
    result = ask(retrieve=FakeRetriever(results=[table]))

    assert ("[S1] file: Annual-Report.pdf | page: 4 | section: Climate > Carbon footprint"
            " | type: table\n") in user.content
    assert "[S2] file: Annual-Report.pdf | page: 6\n" in user.content
    assert result["sources"][0]["chunk_type"] == "table"
    assert result["sources"][0]["section_path"] == "Climate > Carbon footprint"


# ---------- The endpoint ----------

@pytest.fixture
def client(monkeypatch):
    llm = FakeChatModel(answered(source_ids=["S1"]))

    def fake_answer(question, top_k=None, document_id=None, min_similarity=None):
        return answer_question(question, top_k=top_k, document_id=document_id,
                               min_similarity=min_similarity, llm=llm, retrieve=FakeRetriever())
    monkeypatch.setattr(main, "answer_question", fake_answer)
    return TestClient(main.app)


def test_chat_endpoint_returns_structured_json(client):
    response = client.post("/chat", json={"question": "What was the revenue?", "top_k": 3,
                                          "document_id": "doc1"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer"].startswith("Revenue was")
    assert body["sources"][0]["filename"] == "Annual-Report.pdf"
    assert body["sources"][0]["page_number"] == 6
    assert body["sources"][0]["chunk_type"] == "text"
    assert body["citations"] == [{"filename": "Annual-Report.pdf", "pages": [6]}]
    assert body["retrieval"]["top_k"] == 3
    assert body["retrieval"]["document_id"] == "doc1"


@pytest.mark.parametrize("payload", [
    {},
    {"question": "q"},  # document_id is required
    {"question": "", "document_id": "doc1"},
    {"question": "x" * (config.CHAT_MAX_QUESTION_CHARS + 1), "document_id": "doc1"},
    {"question": "q", "document_id": "doc1", "top_k": 0},
    {"question": "q", "document_id": "doc1", "top_k": config.RETRIEVAL_MAX_TOP_K + 1},
    {"question": "q", "document_id": "doc1", "min_similarity": 1.5},
    {"question": "q", "document_id": ""},
    {"question": "q", "document_id": "doc1", "topk": 3},  # misspelled field
])
def test_chat_endpoint_validates_requests(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_chat_endpoint_reports_errors_with_their_status(monkeypatch):
    def failing(*args, **kwargs):
        raise ChatError("The chat model did not respond within 60 seconds.", 504)
    monkeypatch.setattr(main, "answer_question", failing)

    response = TestClient(main.app).post("/chat", json={"question": "q", "document_id": "doc1"})

    assert response.status_code == 504
    assert response.json() == {"detail": "The chat model did not respond within 60 seconds."}


def test_chat_endpoint_returns_404_for_an_unknown_document(monkeypatch):
    llm = FakeChatModel(answered())
    not_found = RetrievalError("Document 'nope' is not in the database.", 404)
    monkeypatch.setattr(main, "answer_question", lambda question, **kwargs: answer_question(
        question, llm=llm, retrieve=FakeRetriever(error=not_found), **kwargs))

    response = TestClient(main.app).post("/chat", json={"question": "q", "document_id": "nope"})

    assert response.status_code == 404
    assert llm.calls == []


def test_chat_endpoint_with_placeholder_credentials(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

    response = TestClient(main.app).post("/chat", json={"question": "What was the revenue?",
                                                        "document_id": "doc1"})

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]
