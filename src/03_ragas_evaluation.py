"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy chính xác từ Bước 2) ─────────────────────────
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

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
if config.PROVIDER == "gemini":
    # Gemini chỉ hỗ trợ một candidate cho mỗi request.
    answer_relevancy.strictness = 1
METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tạo FAISS vectorstore với cấu hình chunk 500/50 như Bước 1."""
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """Chạy một câu hỏi và giữ contexts dưới dạng list[str] cho RAGAS."""
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": ctx_str,
        "question": question,
    })
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """Chạy toàn bộ 50 QA pairs qua prompt version được chỉ định."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    prompt = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy {len(QA_PAIRS)} câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question": qa["question"],
            "reference": qa["reference"],
            "answer": out["answer"],
            "contexts": out["contexts"],
        })
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """Chuyển kết quả RAG thành các SingleTurnSample đúng schema RAGAS."""
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """Đánh giá kết quả bằng bốn metrics và bỏ qua từng giá trị NaN."""
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")
    dataset = build_ragas_dataset(rag_results)
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()
    result = evaluate(
        dataset,
        metrics=METRICS,
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=RunConfig(
            timeout=180,
            max_retries=10,
            max_wait=60,
            max_workers=2,
        ),
    )

    scores = {}
    for metric in METRICS:
        key = metric.name
        raw = np.asarray(result[key], dtype=float)
        if raw.size == 0 or np.isnan(raw).all():
            raise RuntimeError(f"RAGAS metric '{key}' không có score hợp lệ")
        scores[key] = float(np.nanmean(raw))

    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Cần kiểm tra retrieval và prompt bằng một lần chạy API thực tế.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
