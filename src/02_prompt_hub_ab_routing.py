"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "nguyen-hoang-anh-rag-prompt-v1"
PROMPT_V2_NAME = "nguyen-hoang-anh-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI hữu ích và thân thiện. Chỉ trả lời dựa trên context được cung cấp. "
    "Trả lời ngắn gọn trong 2-4 câu. Nếu context không chứa câu trả lời, hãy nói "
    "'Tôi không tìm thấy thông tin này'.\n\nContext:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Chỉ sử dụng các facts trong context được cung cấp. "
    "Trình bày câu trả lời chuyên nghiệp trong 3-5 câu có cấu trúc: kết luận chính, "
    "chi tiết hỗ trợ, rồi mức độ chắc chắn. Không suy đoán; nếu context không chứa câu trả lời, "
    "hãy nói 'Tôi không tìm thấy thông tin này'.\n\nContext:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """Upload hai prompt; coi bản không đổi là trạng thái đồng bộ thành công."""
    prompt_specs = [
        ("V1", PROMPT_V1_NAME, PROMPT_V1, "V1 – ngắn gọn, thân thiện"),
        ("V2", PROMPT_V2_NAME, PROMPT_V2, "V2 – chuyên nghiệp, có cấu trúc"),
    ]
    for label, name, prompt, description in prompt_specs:
        try:
            url = client.push_prompt(
                name,
                object=prompt,
                description=description,
            )
            print(f"✅ Đã push {label} → {url}")
        except Exception as e:
            if "Nothing to commit" in str(e):
                print(f"✅ {label} đã đồng bộ trên Hub (không có thay đổi)")
            else:
                raise RuntimeError(f"Không thể push {label} lên Prompt Hub") from e


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """Pull hai prompt; chỉ dùng template local khi từng lệnh pull thất bại."""
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"⚠️  Pull '{PROMPT_V1_NAME}' lỗi ({e}); dùng local fallback")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"⚠️  Pull '{PROMPT_V2_NAME}' lỗi ({e}); dùng local fallback")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """Ánh xạ request_id ổn định vào V1 hoặc V2 bằng parity của MD5."""
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "rag", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """Retrieve context, tạo answer và trả result đầy đủ để tracing."""
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    context = "\n\n".join(contexts)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question,
    })
    return {
        "question": question,
        "retrieved_contexts": contexts,
        "answer": answer,
        "version": version,
    }


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = setup_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]
        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [prompt-{result['version']}] {question[:55]}...")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
