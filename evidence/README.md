# Evidence Summary

**Sinh viên:** Nguyễn Hoàng Anh — **MSSV:** 2A2202601186 — **Cohort/Khoá:** 4 — **Ngày nộp:** 25/8

## Nhiệm vụ 1 — LangSmith Tracing

- `01_langsmith_traces.png` hiển thị project `day22-lab` với 315 traces trong 7 ngày, vượt yêu cầu tối thiểu 100 traces tổng cộng.

## Nhiệm vụ 2 — Prompt Hub và A/B Routing

- `02_prompt_hub.png` hiển thị hai prompt `nguyen-hoang-anh-rag-prompt-v1` và `nguyen-hoang-anh-rag-prompt-v2`.
- `02_ab_routing_log.txt` ghi đủ 50 truy vấn, gồm V1=19 và V2=31; hai prompt đều được pull từ Hub.

## Nhiệm vụ 3 — RAGAS Evaluation

Cả 50 QA pairs được chạy qua cả V1 và V2. Bốn metrics được tính cho mỗi phiên bản:

| Metric | V1 | V2 | Nhận xét |
|---|---:|---:|---|
| Faithfulness | 0.9708 | 0.9020 | V1 cao hơn; cả hai đều đạt mức thưởng ≥ 0.9 |
| Answer relevancy | 0.9063 | 0.8280 | V1 cao hơn |
| Context recall | 1.0000 | 1.0000 | Hai phiên bản bằng nhau |
| Context precision | 0.9417 | 0.9450 | V2 cao hơn nhẹ |

V1 đạt điểm faithfulness và answer relevancy cao hơn vì prompt yêu cầu câu trả lời ngắn gọn 2–4 câu, nhờ đó hạn chế các chi tiết bổ sung khó đối chiếu với context. V2 có context precision nhỉnh hơn nhưng cấu trúc 3–5 câu làm câu trả lời dài hơn, dẫn đến answer relevancy thấp hơn. Mục tiêu faithfulness ≥ 0.8 đã đạt với cả hai phiên bản.

- `03_ragas_scores.png` ghi lại bảng so sánh trên terminal.
- `03_ragas_report.json` là bản sao của `data/ragas_report.json`.

## Nhiệm vụ 4 — Guardrails Validators

- `04_pii_demo_log.txt` chứng minh email, phone, SSN và credit card được redact; input sạch được giữ nguyên.
- `04_json_demo_log.txt` chứng minh validator xử lý JSON hợp lệ, markdown fences, single quotes, trailing comma và fallback cho input không thể sửa.
