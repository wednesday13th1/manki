from openai import OpenAI
import os, json, re, sys


# ★環境変数でキーを読む（さっき出た sk- のやつ）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0)

PROMPT = """
You are an API that returns study data for English learners.
Return ONLY valid JSON, no prose, no code fence.

Schema (exactly this shape):
{
  "items": [
    {
      "word": "lowercase English word",
      "meaning": "simple English definition",
      "example": "natural sentence using the word",
      "quiz": {
        "question": "same example but the target word replaced by ______",
        "answer": "the target word"
      }
    }
  ]
}

Task:
- Create exactly 10 items for general intermediate vocabulary.
- No extra keys. No explanations. Just the JSON object above.
"""

def extract_json(text: str):
    # フェンス付き/英語文付きの対策（最後の保険）
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("No JSON object found in response")
    return json.loads(m.group(0))

try:
    # JSONを強制：response_format を使う
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # うまくいかないときは "gpt-4o" に変更して試す
        messages=[{"role": "user", "content": PROMPT}],
        response_format={"type": "json_object"},
    )

    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        raise ValueError("Model returned empty content")

    # まず素直にJSONとして読む
    try:
        data = json.loads(raw)
    except Exception:
        data = extract_json(raw)

    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("`items` is missing or not a list")
    if len(items) != 10:
        raise ValueError(f"`items` must have 10 elements, got {len(items)}")

    # ざっくりキー確認
    for i, it in enumerate(items, 1):
        for k in ("word", "meaning", "example", "quiz"):
            if k not in it:
                raise ValueError(f"item {i} missing key: {k}")
        if not isinstance(it["quiz"], dict) or "question" not in it["quiz"] or "answer" not in it["quiz"]:
            raise ValueError(f"item {i} quiz invalid")

    print(json.dumps(items, indent=2, ensure_ascii=False))
    print("\n✅ OK — 10件の単語を取得しました。")

except Exception as e:
    # デバッグしやすいように生返答も表示
    print("ERROR:", type(e).__name__, e, file=sys.stderr)
    print("\n--- raw content from model ---\n", raw if 'raw' in locals() else "(no content)", file=sys.stderr)
    sys.exit(1)
