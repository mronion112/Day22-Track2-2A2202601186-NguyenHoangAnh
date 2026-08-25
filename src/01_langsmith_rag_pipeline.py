"""
Bước 1 — RAG Pipeline với LangSmith Tracing
=============================================
NHIỆM VỤ:
  1. Tải knowledge base, chia chunks, index với FAISS
  2. Xây dựng RAG chain: retriever → prompt → LLM → output parser
  3. Trang trí hàm query với @traceable để LangSmith ghi lại mỗi lần gọi
  4. Chạy 50 câu hỏi → tạo ≥ 50 traces trên LangSmith

DELIVERABLE: Mở https://smith.langchain.com → project của bạn → xác nhận ≥ 50 traces.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ⚠️ QUAN TRỌNG: Import config TRƯỚC KHI import bất kỳ thư viện LangChain nào.
# config.py đồng bộ các biến tracing canonical LANGSMITH_* và alias LANGCHAIN_*.
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Thiết lập Vectorstore ───────────────────────────────────────────────
def setup_vectorstore():
    """Tải knowledge base, chia chunks 500/50 và tạo FAISS vectorstore."""
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"📚 Đã chia thành {len(chunks)} chunks")
    return build_vectorstore(chunks, embeddings)


# ── 2. RAG Prompt Template ─────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Bạn là trợ lý AI hữu ích. Chỉ trả lời dựa trên context được cung cấp. "
        "Nếu context không chứa câu trả lời, hãy nói "
        "'Tôi không tìm thấy thông tin này'.\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])


# ── 3. Build RAG Chain ─────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """Xây dựng LCEL chain và retriever lấy ba tài liệu gần nhất."""
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 4. Hàm Query có LangSmith Tracing ─────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """Chạy RAG chain; decorator ghi input, output và latency vào LangSmith."""
    return chain.invoke(question)


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()
    chain, _retriever = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {str(answer)[:100]}\n")

    print(f"\n✅ {len(SAMPLE_QUESTIONS)} traces đã gửi lên LangSmith project '{config.LANGSMITH_PROJECT}'")
    print("   Mở https://smith.langchain.com để xem traces.")


if __name__ == "__main__":
    main()
