"""
Bước 4 — Guardrails AI Validators
====================================
NHIỆM VỤ:
  1. Xây dựng PIIDetector: phát hiện & redact email, số điện thoại, SSN, số thẻ tín dụng
  2. Xây dựng JSONFormatter: tự động sửa JSON lỗi
  3. Bọc mỗi validator trong Guard và test với các mẫu đầu vào
  4. Chạy demo với 6 trường hợp PII và 5 trường hợp JSON

DELIVERABLE: Tất cả test cases pass (PII bị redact, JSON được sửa thành công)

CÁC KHÁI NIỆM CHÍNH:
  - @register_validator     — khai báo custom validator class
  - Validator.validate()    — implement logic kiểm tra + sửa
  - OnFailAction.FIX        — thay thế output thay vì raise error
  - Guard().use(validator)  — gắn validator instance vào guard
  - guard.validate(text)    → ValidationOutcome
      .validation_passed    — bool
      .validated_output     — output đã được xử lý

⚠️  QUAN TRỌNG: on_fail phải truyền vào CONSTRUCTOR của VALIDATOR, KHÔNG phải Guard.use()
    SAI  : Guard().use(PIIDetector, on_fail=OnFailAction.FIX)   ← TypeError
    ĐÚNG : Guard().use(PIIDetector(on_fail=OnFailAction.FIX))   ← correct
"""

import argparse
import re
import json

from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult

try:
    from guardrails.hub import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction


# ── 1. PII Detector Validator ──────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """Phát hiện và redact email, điện thoại, SSN và số thẻ tín dụng."""

    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE":       r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\w)",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        """Trả FailResult có fix_value đã redact để OnFailAction.FIX áp dụng."""
        redacted_text = value
        found_pii = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, redacted_text)
            if matches:
                found_pii.extend([pii_type] * len(matches))
                redacted_text = re.sub(
                    pattern,
                    f"[{pii_type}_REDACTED]",
                    redacted_text,
                )

        if found_pii:
            print(f"  ⚠️  Đã redact {len(found_pii)} PII: {found_pii}")
            return FailResult(
                error_message=f"Phát hiện PII: {', '.join(found_pii)}",
                fix_value=redacted_text,
            )

        return PassResult()


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Parse JSON và tự động sửa fences, single quotes, trailing commas."""

    @staticmethod
    def _repair(text: str) -> str:
        """Sửa các lỗi JSON phổ biến mà không gọi LLM hay dịch vụ ngoài."""
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text).strip()
        text = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', text)
        text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
        return re.sub(r",\s*([}\]])", r"\1", text)

    def validate(self, value: str, metadata: dict):
        """Trả JSON chuẩn; repaired/fallback đi qua FailResult.fix_value."""
        try:
            parsed = json.loads(value)
            return PassResult(
                value_override=json.dumps(parsed, ensure_ascii=False, indent=2)
            )
        except json.JSONDecodeError:
            pass

        try:
            parsed = json.loads(self._repair(value))
            fixed = json.dumps(parsed, ensure_ascii=False, indent=2)
            print("  🔧 JSON đã được sửa thành công")
            return FailResult(
                error_message="JSON đầu vào cần được sửa định dạng",
                fix_value=fixed,
            )
        except json.JSONDecodeError as e:
            fallback = json.dumps(
                {
                    "error": "Không thể phân tích JSON",
                    "raw": value[:200],
                },
                ensure_ascii=False,
                indent=2,
            )
            return FailResult(
                error_message=f"JSON không hợp lệ sau khi sửa: {e}",
                fix_value=fallback,
            )


# ── 3. Demo: PII Guard ─────────────────────────────────────────────────────
def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  Demo: PII Detection & Redaction")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Email",        "Contact John at john.doe@example.com for details."),
        ("Phone",        "Call our support line at (555) 867-5309."),
        ("SSN",          "Patient SSN is 123-45-6789 on file."),
        ("Credit Card",  "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",    "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",        "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "FIX" if result.validated_output != text else "PASS"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 4. Demo: JSON Guard ────────────────────────────────────────────────────
def demo_json_guard():
    print("\n" + "=" * 55)
    print("  Demo: JSON Formatting & Repair")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Valid JSON",       '{"name": "Alice", "age": 30}'),
        ("Markdown fences",  '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",    "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",   '{"key": "value",}'),
        ("Truly invalid",    "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "FIX" if result.validated_output != text else "PASS"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {result.validated_output}")


# ── 5. Main ────────────────────────────────────────────────────────────────
def main(demo: str = "all"):
    print("=" * 55)
    print("  Bước 4: Guardrails AI Validators")
    print("=" * 55)

    if demo in ("pii", "all"):
        demo_pii_guard()
    if demo in ("json", "all"):
        demo_json_guard()

    print("\n✅ Bước 4 hoàn thành!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy demo Guardrails local")
    parser.add_argument(
        "--demo",
        choices=["pii", "json", "all"],
        default="all",
        help="Chọn demo cần chạy (mặc định: all)",
    )
    main(parser.parse_args().demo)
