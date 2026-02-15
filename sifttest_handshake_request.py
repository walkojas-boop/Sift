from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _canonical_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _build_signature(
    *,
    shared_key: str,
    agent_id: str,
    method: str,
    path: str,
    nonce: str,
    timestamp: int,
    body: dict,
) -> str:
    body_hash = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    material = "|".join([agent_id, method.upper(), path, str(timestamp), nonce, body_hash])
    return hmac.new(shared_key.encode("utf-8"), material.encode("utf-8"), hashlib.sha256).hexdigest()


def _load_intake_token(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing intake token file: {path}")
    raw = json.loads(p.read_text(encoding="utf-8-sig"))
    token = raw.get("token") if isinstance(raw, dict) else None
    if not isinstance(token, str) or not token.strip():
        raise ValueError("intake token file must contain a non-empty 'token' field")
    return token.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and execute a signed first-handshake request for SiftTest")
    parser.add_argument("--url", default="http://127.0.0.1:8000/request")
    parser.add_argument("--path", default="/request")
    parser.add_argument("--agent-id", default="SiftTest")
    parser.add_argument("--shared-key", default="sifttest_shared_hmac_key_v1")
    parser.add_argument("--token-file", default="sifttest_intake_token.json")
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--print-only", action="store_true", help="Print headers/body without sending request")
    args = parser.parse_args()

    payload = {
        "agent_handshake": {
            "agent_id": args.agent_id,
            "response_format": "json",
        },
        "execution_payload": {
            "intent": "Translate this sentence to Spanish: We are ready for alpha validation.",
            "constraints": {
                "budget": 0.0,
                "latency_max": 1000,
                "min_certainty": 4.0,
            },
        },
    }

    nonce = secrets.token_hex(16)
    timestamp = int(time.time())
    signature = _build_signature(
        shared_key=args.shared_key,
        agent_id=args.agent_id,
        method="POST",
        path=args.path,
        nonce=nonce,
        timestamp=timestamp,
        body=payload,
    )
    auth_header = f"nonce={nonce},timestamp={timestamp},signature={signature}"
    intake_token = _load_intake_token(args.token_file)

    print("Sift-Auth-Header:", auth_header)
    print("Sift-Intake-Token:", intake_token)
    print("Payload:", json.dumps(payload, indent=2, sort_keys=True))

    if args.print_only:
        return 0

    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = Request(
        args.url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Sift-Auth-Header": auth_header,
            "Sift-Intake-Token": intake_token,
        },
    )

    try:
        with urlopen(req, timeout=args.timeout_seconds) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            print("HTTP", resp.status)
            print(text)
            return 0
    except HTTPError as exc:
        print("HTTP", exc.code)
        print(exc.read().decode("utf-8", errors="replace"))
        return 1
    except URLError as exc:
        print("NETWORK_ERROR", str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
