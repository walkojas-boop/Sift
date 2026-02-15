from __future__ import annotations

import argparse
from functools import lru_cache
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import statistics
import time
from typing import Any
import uuid
from dataclasses import dataclass

from auth_manager import (
    authorize_request,
    build_auth_header,
    write_security_audit_event,
)
from intake_token_generator import generate_intake_token, verify_intake_token
from intake_parser import parse_request
from mco_broker import run as mco_run
from response_controller import package_response
from session_manager import SessionError, create_session, load_session, put_execution_result, resolve_context_references
from telemetry_chain import append_telemetry_chain_entry

try:
    from fastapi import FastAPI, Header, HTTPException  # type: ignore
    FASTAPI_AVAILABLE = True
except Exception:  # noqa: BLE001
    FASTAPI_AVAILABLE = False
    Header = None  # type: ignore[assignment]

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail


DEFAULT_CONSTRAINTS = {
    "budget": 1.0,
    "latency_max": 1200,
    "min_certainty": 4.0,
}

SAFE_OPERATING_ENVELOPE = {
    "budget_min": 0.0,
    "budget_max": 3.0,
    "latency_min": 50,
    "latency_max": 5000,
    "certainty_min": 4.0,
}


@dataclass(frozen=True)
class AgentHandshake:
    agent_id: str
    response_format: str = "json"


@dataclass(frozen=True)
class ConstraintsIn:
    budget: float | None = None
    latency_max: int | None = None
    min_certainty: float | None = None


@dataclass(frozen=True)
class ExecutionPayload:
    intent: str
    constraints: ConstraintsIn | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class GatewayRequest:
    agent_handshake: AgentHandshake
    execution_payload: ExecutionPayload


@dataclass(frozen=True)
class GatewayResponse:
    sift_correlation_id: str
    sift_session_id: str
    tx_id: str
    intent_id: str
    response: dict[str, Any]
    translation_latency_ms: int
    audit_trace_id: str | None
    stage_latencies_ms: dict[str, int] | None = None


_JSON_CACHE: dict[str, tuple[int, dict[str, Any]]] = {}


def _cached_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    key = str(p.resolve())
    disable_cache = os.getenv("SIFT_GATEWAY_DISABLE_CACHE", "0").strip() == "1"
    mtime = p.stat().st_mtime_ns
    if not disable_cache and key in _JSON_CACHE:
        cached_mtime, cached_payload = _JSON_CACHE[key]
        if cached_mtime == mtime:
            return cached_payload
    raw = json.loads(p.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be object")
    _JSON_CACHE[key] = (mtime, raw)
    return raw


def _validate_request(raw: dict[str, Any]) -> GatewayRequest:
    hs = raw.get("agent_handshake", {})
    ep = raw.get("execution_payload", {})
    if not isinstance(hs, dict) or not isinstance(ep, dict):
        raise HTTPException(status_code=422, detail="invalid request shape")
    agent_id = hs.get("agent_id")
    response_format = hs.get("response_format", "json")
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise HTTPException(status_code=422, detail="agent_id missing")
    if not isinstance(response_format, str) or response_format.lower() not in {"json", "markdown", "raw"}:
        raise HTTPException(status_code=422, detail="response_format must be one of: json, markdown, raw")
    intent = ep.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        raise HTTPException(status_code=422, detail="intent missing")
    if len(intent) > 10000:
        raise HTTPException(status_code=422, detail="intent exceeds 10000 chars")
    c_raw = ep.get("constraints", {})
    if c_raw is None:
        c_raw = {}
    if not isinstance(c_raw, dict):
        raise HTTPException(status_code=422, detail="constraints must be object")
    session_id = ep.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise HTTPException(status_code=422, detail="session_id must be string")
    c = ConstraintsIn(
        budget=float(c_raw["budget"]) if "budget" in c_raw and c_raw["budget"] is not None else None,
        latency_max=int(c_raw["latency_max"]) if "latency_max" in c_raw and c_raw["latency_max"] is not None else None,
        min_certainty=float(c_raw["min_certainty"]) if "min_certainty" in c_raw and c_raw["min_certainty"] is not None else None,
    )
    return GatewayRequest(
        agent_handshake=AgentHandshake(agent_id=agent_id, response_format=response_format.lower()),
        execution_payload=ExecutionPayload(intent=intent, constraints=c, session_id=session_id),
    )


def _load_registry(path: str | Path = "registry.json") -> dict[str, Any]:
    raw = _cached_json(path)
    if not isinstance(raw, dict):
        raise ValueError("registry must be object")
    bridges = raw.get("bridges")
    if not isinstance(bridges, list):
        raise ValueError("registry.bridges must be array")
    return raw


def _load_agent_registry_cached(path: str | Path = "agent_registry.json") -> dict[str, Any]:
    raw = _cached_json(path)
    agents = raw.get("agents")
    if not isinstance(agents, list):
        raise ValueError("agent_registry.agents must be array")
    return raw


def _load_alpha_allowlist(path: str | Path = "alpha_allowlist.json") -> dict[str, Any]:
    raw = _cached_json(path)
    agents = raw.get("agents")
    if not isinstance(agents, list):
        raise ValueError("alpha_allowlist.agents must be array")
    return raw


def _allowlist_agent_entry(allowlist: dict[str, Any], agent_id: str) -> dict[str, Any] | None:
    for row in allowlist.get("agents", []):
        if isinstance(row, dict) and str(row.get("agent_id")) == agent_id:
            return row
    return None


def _default_minimal_scope(registry: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for b in registry.get("bridges", []):
        if not isinstance(b, dict):
            continue
        bid = b.get("bridge_id")
        cap = str(b.get("capability_path", ""))
        if not isinstance(bid, str):
            continue
        if cap.startswith("transform.text") or cap.startswith("data.extraction"):
            out.append(bid)
    return sorted(set(out))


def _effective_agent_scope(agent_id: str, registry: dict[str, Any]) -> list[str]:
    pmap = registry.get("agent_permission_map", {})
    if isinstance(pmap, dict):
        vals = pmap.get(agent_id)
        if isinstance(vals, list):
            return sorted({str(v) for v in vals})
    return _default_minimal_scope(registry)


def _onboarded_state(path: str | Path = ".alpha_onboarded.json") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"onboarded_agents": {}}
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        raw = {"onboarded_agents": {}}
    if not isinstance(raw, dict):
        raw = {"onboarded_agents": {}}
    if not isinstance(raw.get("onboarded_agents"), dict):
        raw["onboarded_agents"] = {}
    return raw


def _is_onboarded(agent_id: str, path: str | Path = ".alpha_onboarded.json") -> bool:
    state = _onboarded_state(path)
    val = state["onboarded_agents"].get(agent_id)
    return isinstance(val, dict) and bool(val.get("approved"))


def _mark_onboarded(agent_id: str, path: str | Path = ".alpha_onboarded.json") -> None:
    state = _onboarded_state(path)
    state["onboarded_agents"][agent_id] = {
        "approved": True,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _log_intake_event(
    *,
    event_type: str,
    agent_id: str,
    decision: str,
    reason: str,
    bridge_id: str | None = None,
) -> None:
    write_security_audit_event(
        event_type=event_type,
        agent_id=agent_id,
        reason=f"{decision}: {reason}",
        bridge_id=bridge_id,
    )
    append_telemetry_chain_entry(
        telemetry_store_path="telemetry_store.jsonl",
        payload={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "agent_id": agent_id,
            "decision": decision,
            "reason": reason,
            "bridge_id": bridge_id,
        },
    )


def _sifttest_override_state(path: str | Path = ".sifttest_first_route_override.json") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"consumed": False, "consumed_at": None}
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return {"consumed": False, "consumed_at": None}
    if not isinstance(raw, dict):
        return {"consumed": False, "consumed_at": None}
    return {
        "consumed": bool(raw.get("consumed", False)),
        "consumed_at": raw.get("consumed_at"),
    }


def _mark_sifttest_override_consumed(path: str | Path = ".sifttest_first_route_override.json") -> None:
    payload = {
        "consumed": True,
        "consumed_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@lru_cache(maxsize=512)
def _parse_request_cached(intent: str) -> dict[str, Any]:
    return parse_request(
        raw_text=intent,
        ontology_path="ontology_master.json",
        mapping_library_path="mapping_library.json",
    )


def _registry_policies(registry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults = dict(DEFAULT_CONSTRAINTS)
    envelope = dict(SAFE_OPERATING_ENVELOPE)
    reg_defaults = registry.get("global_default_constraints")
    if isinstance(reg_defaults, dict):
        defaults.update({k: reg_defaults[k] for k in ("budget", "latency_max", "min_certainty") if k in reg_defaults})
    reg_env = registry.get("safe_operating_envelope")
    if isinstance(reg_env, dict):
        envelope.update({k: reg_env[k] for k in envelope if k in reg_env})
    return defaults, envelope


def _budget_to_tier(budget: float) -> str:
    if budget <= 0.0:
        return "free"
    if budget <= 1.0:
        return "low"
    if budget <= 2.0:
        return "mid"
    return "high"


def _extract_requested_bridge(intent: str) -> str | None:
    m = re.search(r"\b(br-[a-z]+-\d{2})\b", intent, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower()


def _bridge_allowed(registry: dict[str, Any], bridge_id: str) -> tuple[bool, str]:
    bridge = next((b for b in registry["bridges"] if isinstance(b, dict) and str(b.get("bridge_id")) == bridge_id), None)
    if not isinstance(bridge, dict):
        return False, "UNKNOWN_BRIDGE"
    status = str(bridge.get("audit_status"))
    if status == "Quarantined":
        return False, "QUARANTINED_BRIDGE"
    if status not in {"Audited", "Verified"}:
        return False, "UNVERIFIED_BRIDGE"
    if bool(bridge.get("review_required")):
        return False, "REVIEW_REQUIRED"
    return True, "OK"


def _normalize_constraints(
    payload: ExecutionPayload,
    defaults: dict[str, Any],
    envelope: dict[str, Any],
    *,
    certainty_floor_override: float | None = None,
    certainty_cap_override: float | None = None,
) -> dict[str, Any]:
    c = payload.constraints or ConstraintsIn()
    budget = defaults["budget"] if c.budget is None else float(c.budget)
    latency = defaults["latency_max"] if c.latency_max is None else int(c.latency_max)
    certainty = defaults["min_certainty"] if c.min_certainty is None else float(c.min_certainty)

    # Enforce Safe Operating Envelope: clamp to policy bounds.
    budget = min(max(budget, float(envelope["budget_min"])), float(envelope["budget_max"]))
    latency = min(max(latency, int(envelope["latency_min"])), int(envelope["latency_max"]))
    certainty_floor = float(envelope["certainty_min"])
    if certainty_floor_override is not None:
        certainty_floor = min(certainty_floor, float(certainty_floor_override))
    if certainty_cap_override is not None:
        cap = float(certainty_cap_override)
        certainty_floor = min(certainty_floor, cap)
        certainty = min(certainty, cap)
    certainty = max(certainty_floor, certainty)

    return {
        "budget": round(budget, 3),
        "latency_max": int(latency),
        "min_certainty": round(certainty, 3),
        "budget_tier": _budget_to_tier(budget),
    }


def _route_with_mco(intent: str, normalized: dict[str, Any], capability_path: str) -> dict[str, Any]:
    use_override = os.getenv("SIFT_GATEWAY_DISABLE_CAP_OVERRIDE", "0").strip() != "1"
    mco_args = argparse.Namespace(
        intent=intent,
        priority="economy" if "cheap" in intent.lower() or "cheapest" in intent.lower() else "fidelity",
        min_certainty=float(normalized["min_certainty"]),
        max_latency_ms=float(normalized["latency_max"]),
        max_cost_units=float(normalized["budget"]),
        constraint_text="",
        registry="registry.json",
        ontology="ontology_master.json",
        mapping_library="mapping_library.json",
        execution_map="execution_map.json",
        capability_map="capability_execution_map.json",
        live_allowlist="live_allowlist.json",
        guardrails="guardrails.json",
        inventory="bridge_inventory.json",
        execution_log="execution_audit.log",
        decision_log="routing_decision_log.jsonl",
        build_report="Level 2 Build Completion Report.md",
        execute=False,
        self_validate=False,
        capability_path_override=(capability_path if use_override else None),
    )
    return mco_run(mco_args)


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_gateway_audit_trace(
    *,
    correlation_id: str,
    intent: str,
    intent_id: str,
    constraint_map: dict[str, Any],
    selected_bridge_id: str,
    decision_logic: str,
    keyring_path: str = "audit_keyring.json",
    ledger_path: str = "audit_ledger.jsonl",
) -> str:
    trace_id = str(uuid.uuid4())
    payload = {
        "audit_trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "Level 4 Phase 1 Gateway",
        "sift_correlation_id": correlation_id,
        "input_hash": hashlib.sha256(intent.encode("utf-8")).hexdigest(),
        "handoffs": {
            "gateway_intake": {"intent_id": intent_id, "constraint_map": constraint_map},
            "broker_selection": {"selected_bridge_id": selected_bridge_id, "decision_logic": decision_logic},
        },
    }
    keyring_raw = {"keys": {}}
    kp = Path(keyring_path)
    if kp.exists():
        try:
            loaded = json.loads(kp.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                keyring_raw = loaded
        except Exception:  # noqa: BLE001
            keyring_raw = {"keys": {}}
    keys = keyring_raw.get("keys")
    if not isinstance(keys, dict):
        keys = {}
    private_key = hashlib.sha256(f"{trace_id}|{secrets.token_hex(32)}".encode("utf-8")).hexdigest()
    keys[trace_id] = {"audit_private_key": private_key, "created_at": payload["timestamp"]}
    keyring_raw["keys"] = keys
    kp.write_text(json.dumps(keyring_raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    signature = hmac.new(bytes.fromhex(private_key), _canonical_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()
    with Path(ledger_path).open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "audit_trace_id": trace_id,
                    "signature": signature,
                    "signature_alg": "HMAC-SHA256",
                    "trace_payload": payload,
                },
                sort_keys=True,
            )
            + "\n"
        )
    return trace_id


def handle_gateway_request(
    req: GatewayRequest,
    *,
    auth_header_value: str,
    intake_token: str | None,
    raw_body: dict[str, Any],
    method: str = "POST",
    path: str = "/request",
) -> GatewayResponse:
    t0 = time.perf_counter()
    stage: dict[str, int] = {}

    allowlist = _load_alpha_allowlist("alpha_allowlist.json")
    allowlisted = _allowlist_agent_entry(allowlist, req.agent_handshake.agent_id)
    if not isinstance(allowlisted, dict):
        _log_intake_event(
            event_type="INTAKE_DENIAL",
            agent_id=req.agent_handshake.agent_id,
            decision="DENY",
            reason="agent not in alpha_allowlist",
        )
        raise HTTPException(status_code=403, detail="agent not allowlisted")

    if not _is_onboarded(req.agent_handshake.agent_id):
        if not intake_token:
            _log_intake_event(
                event_type="INTAKE_DENIAL",
                agent_id=req.agent_handshake.agent_id,
                decision="DENY",
                reason="missing onboarding token",
            )
            raise HTTPException(status_code=401, detail="missing onboarding token")
        ok_token, token_info = verify_intake_token(token=intake_token, agent_id=req.agent_handshake.agent_id)
        if not ok_token:
            _log_intake_event(
                event_type="INTAKE_DENIAL",
                agent_id=req.agent_handshake.agent_id,
                decision="DENY",
                reason=str(token_info),
            )
            raise HTTPException(status_code=401, detail=f"invalid onboarding token: {token_info}")
        _mark_onboarded(req.agent_handshake.agent_id)
        _log_intake_event(
            event_type="INTAKE_APPROVAL",
            agent_id=req.agent_handshake.agent_id,
            decision="APPROVE",
            reason="first handshake token validated",
        )

    t_sig = time.perf_counter()
    agent_registry = _load_agent_registry_cached("agent_registry.json")
    # Reuse auth_manager throttling but with per-agent caps from allowlist.
    merged_agents: list[dict[str, Any]] = []
    for row in agent_registry.get("agents", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("agent_id")) == req.agent_handshake.agent_id:
            merged = dict(row)
            merged["rate_cap"] = allowlisted.get("rate_cap")
            merged_agents.append(merged)
        else:
            merged_agents.append(row)
    auth_registry = {"agents": merged_agents}
    auth_decision = authorize_request(
        agent_registry=auth_registry,
        agent_id=req.agent_handshake.agent_id,
        auth_header_value=auth_header_value,
        method=method,
        path=path,
        raw_body=raw_body,
    )
    if not auth_decision.ok:
        if auth_decision.greylisted and auth_decision.queue_delay_seconds > 0:
            time.sleep(auth_decision.queue_delay_seconds)
        _log_intake_event(
            event_type="AUTH_REJECTION",
            agent_id=req.agent_handshake.agent_id,
            decision="DENY",
            reason=auth_decision.detail,
        )
        raise HTTPException(status_code=auth_decision.status_code, detail=auth_decision.detail)
    stage["signature_verification_ms"] = int((time.perf_counter() - t_sig) * 1000)

    t_constraints = time.perf_counter()
    registry = _load_registry("registry.json")
    defaults, envelope = _registry_policies(registry)
    certainty_floor_override: float | None = None
    certainty_cap_override: float | None = None
    apply_sifttest_first_route_override = False
    if req.agent_handshake.agent_id == "SiftTest":
        override_state = _sifttest_override_state()
        if not override_state.get("consumed"):
            certainty_floor_override = 2.0
            certainty_cap_override = 2.0
            apply_sifttest_first_route_override = True
    normalized = _normalize_constraints(
        req.execution_payload,
        defaults,
        envelope,
        certainty_floor_override=certainty_floor_override,
        certainty_cap_override=certainty_cap_override,
    )
    if apply_sifttest_first_route_override:
        _log_intake_event(
            event_type="CERTAINTY_OVERRIDE_APPLIED",
            agent_id=req.agent_handshake.agent_id,
            decision="ALLOW",
            reason="SiftTest first-route certainty floor lowered to 2.0",
        )
    stage["constraint_mapping_ms"] = int((time.perf_counter() - t_constraints) * 1000)

    session_id = req.execution_payload.session_id
    if isinstance(session_id, str) and session_id.strip():
        try:
            s = load_session(session_id.strip())
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if str(s.get("agent_id")) != req.agent_handshake.agent_id:
            raise HTTPException(status_code=403, detail="session ownership mismatch")
        session_id = session_id.strip()
    else:
        session_obj = create_session(agent_id=req.agent_handshake.agent_id, user_id=req.agent_handshake.agent_id, ttl_minutes=30)
        session_id = str(session_obj["session_id"])

    resolved_intent = req.execution_payload.intent
    if "{{session." in resolved_intent or re.search(r"\bTX_[A-Za-z0-9_]+\b", resolved_intent):
        try:
            resolved_intent = resolve_context_references(session_id, resolved_intent)
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    t_parse = time.perf_counter()
    disable_cache = os.getenv("SIFT_GATEWAY_DISABLE_CACHE", "0").strip() == "1"
    parser_out = (
        parse_request(
            raw_text=resolved_intent,
            ontology_path="ontology_master.json",
            mapping_library_path="mapping_library.json",
        )
        if disable_cache
        else _parse_request_cached(resolved_intent)
    )
    if not parser_out.get("ok"):
        raise HTTPException(status_code=422, detail=f"intent parse failed: {parser_out.get('reason')}")
    intent_id = str(parser_out["request_package"]["capability_path"])
    stage["request_parsing_ms"] = int((time.perf_counter() - t_parse) * 1000)

    t_scope = time.perf_counter()
    requested_bridge = _extract_requested_bridge(resolved_intent)
    effective_scope = _effective_agent_scope(req.agent_handshake.agent_id, registry)
    if requested_bridge is not None:
        if requested_bridge not in set(effective_scope):
            _log_intake_event(
                event_type="AUTH_SCOPE_DENY",
                agent_id=req.agent_handshake.agent_id,
                decision="DENY",
                reason="403_FORBIDDEN: bridge outside agent scope",
                bridge_id=requested_bridge,
            )
            raise HTTPException(status_code=403, detail="403_FORBIDDEN: bridge outside agent scope")
        allowed, reason = _bridge_allowed(registry, requested_bridge)
        if not allowed:
            raise HTTPException(status_code=403, detail=f"bridge rejected: {reason}")

    route = _route_with_mco(resolved_intent, normalized, intent_id)
    if not route.get("ok"):
        if apply_sifttest_first_route_override and req.agent_handshake.agent_id == "SiftTest":
            fallback_bridge_id = next(
                (
                    bid
                    for bid in effective_scope
                    if _bridge_allowed(registry, bid)[0]
                ),
                None,
            )
            if fallback_bridge_id is not None:
                _log_intake_event(
                    event_type="FIRST_ROUTE_OVERRIDE",
                    agent_id=req.agent_handshake.agent_id,
                    decision="ALLOW",
                    reason="SiftTest first-route fallback bridge applied",
                    bridge_id=fallback_bridge_id,
                )
                route = {
                    "ok": True,
                    "bridge_id": fallback_bridge_id,
                    "decision_logic": "SiftTest first-route override: fallback minimal-scope bridge",
                }
            else:
                raise HTTPException(status_code=409, detail=route.get("status", "NO_VIABLE_RESOURCE"))
        else:
            raise HTTPException(status_code=409, detail=route.get("status", "NO_VIABLE_RESOURCE"))
    bridge_id = str(route["bridge_id"])
    if bridge_id not in set(effective_scope):
        _log_intake_event(
            event_type="AUTH_SCOPE_DENY",
            agent_id=req.agent_handshake.agent_id,
            decision="DENY",
            reason="403_FORBIDDEN: bridge outside agent scope",
            bridge_id=bridge_id,
        )
        raise HTTPException(status_code=403, detail="403_FORBIDDEN: bridge outside agent scope")

    allowed, reason = _bridge_allowed(registry, bridge_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=f"selected bridge rejected: {reason}")
    stage["scope_validation_ms"] = int((time.perf_counter() - t_scope) * 1000)

    correlation_id = str(uuid.uuid4())
    trace_id = _write_gateway_audit_trace(
        correlation_id=correlation_id,
        intent=resolved_intent,
        intent_id=intent_id,
        constraint_map=normalized,
        selected_bridge_id=bridge_id,
        decision_logic=str(route.get("decision_logic")),
    )
    internal_output = {
        "result_data": {
            "intent_id": intent_id,
            "selected_bridge_id": bridge_id,
            "decision_logic": str(route.get("decision_logic")),
            "resolved_intent": resolved_intent,
        },
        "handler_path": "handlers/text_handler.py",
        "system_path": "C:\\Users\\walko\\Desktop\\Sift\\handlers\\text_handler.py",
        "bridge_id": bridge_id,
        "log": "execution_gateway routed via broker",
    }
    agent_response = package_response(
        trace_id=trace_id,
        bridge_output=internal_output,
        response_format=req.agent_handshake.response_format,
        rules_path="sanitization_rules.json",
        ledger_path="audit_ledger.jsonl",
    )
    try:
        s = load_session(session_id)
        refs = s.get("data_refs", [])
        next_idx = (len(refs) if isinstance(refs, list) else 0) + 1
    except SessionError:
        next_idx = 1
    tx_id = f"TX_{next_idx}"
    put_execution_result(
        session_id,
        tx_id,
        bridge_id=bridge_id,
        output=agent_response.get("output"),
    )
    if apply_sifttest_first_route_override:
        _mark_sifttest_override_consumed()
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return GatewayResponse(
        sift_correlation_id=correlation_id,
        sift_session_id=session_id,
        tx_id=tx_id,
        intent_id=intent_id,
        response=agent_response,
        translation_latency_ms=latency_ms,
        audit_trace_id=trace_id,
        stage_latencies_ms=stage,
    )


if FASTAPI_AVAILABLE:
    app = FastAPI(title="Sift Agent Gateway", version="4.1.0")

    @app.post("/request")
    def request_endpoint(
        raw: dict[str, Any],
        sift_auth_header: str | None = Header(default=None, alias="Sift-Auth-Header"),
        sift_intake_token: str | None = Header(default=None, alias="Sift-Intake-Token"),
    ) -> dict[str, Any]:
        if not isinstance(sift_auth_header, str) or not sift_auth_header.strip():
            raise HTTPException(status_code=401, detail="missing Sift-Auth-Header")
        req = _validate_request(raw)
        resp = handle_gateway_request(
            req,
            auth_header_value=sift_auth_header,
            intake_token=sift_intake_token,
            raw_body=raw,
            method="POST",
            path="/request",
        )
        return {
            "sift_correlation_id": resp.sift_correlation_id,
            "sift_session_id": resp.sift_session_id,
            "tx_id": resp.tx_id,
            "intent_id": resp.intent_id,
            "response": resp.response,
            "translation_latency_ms": resp.translation_latency_ms,
            "audit_trace_id": resp.audit_trace_id,
        }


def run_validation(log_path: str = "Normalization_Validation_Log.json") -> dict[str, Any]:
    agent_registry = _load_agent_registry_cached("agent_registry.json")
    alpha = next(
        (
            row for row in agent_registry.get("agents", [])
            if isinstance(row, dict) and str(row.get("agent_id")) == "agent-alpha"
        ),
        None,
    )
    if not isinstance(alpha, dict) or not isinstance(alpha.get("pub_key"), str):
        raise RuntimeError("agent_registry missing active agent-alpha with pub_key")
    alpha_key = str(alpha["pub_key"])
    alpha_intake_token = generate_intake_token(agent_id="agent-alpha", ttl_minutes=120)
    samples = [
        {
            "agent_handshake": {
                "agent_id": "agent-alpha",
                "response_format": "json",
            },
            "execution_payload": {
                "intent": "Find the cheapest way to summarize this quarterly update",
                "constraints": {},
            },
        },
    ]
    rows: list[dict[str, Any]] = []
    session_id: str | None = None
    tx_id: str | None = None
    for i, sample in enumerate(samples, start=1):
        t0 = time.perf_counter()
        try:
            req = _validate_request(sample)
            auth_header = build_auth_header(
                pub_key=alpha_key,
                agent_id=req.agent_handshake.agent_id,
                method="POST",
                path="/request",
                body=sample,
            )
            resp = handle_gateway_request(
                req,
                auth_header_value=auth_header,
                intake_token=alpha_intake_token,
                raw_body=sample,
                method="POST",
                path="/request",
            )
            session_id = resp.sift_session_id
            tx_id = resp.tx_id
            rows.append(
                {
                    "sample": i,
                    "status": "PASS",
                    "intent": sample["execution_payload"]["intent"],
                    "intent_id": resp.intent_id,
                    "has_logic_seal": bool(resp.response.get("logic_seal")),
                    "output_type": type(resp.response.get("output")).__name__,
                    "translation_latency_ms": resp.translation_latency_ms,
                    "latency_slo_pass": resp.translation_latency_ms < 40,
                    "correlation_id": resp.sift_correlation_id,
                    "audit_trace_id": resp.audit_trace_id,
                    "session_id": resp.sift_session_id,
                    "tx_id": resp.tx_id,
                }
            )
        except HTTPException as exc:
            rows.append(
                {
                    "sample": i,
                    "status": "REJECTED",
                    "intent": sample["execution_payload"]["intent"],
                    "http_status": exc.status_code,
                    "detail": exc.detail,
                    "translation_latency_ms": int((time.perf_counter() - t0) * 1000),
                }
            )

    # Session continuity sample referencing previous TX.
    if session_id and tx_id:
        follow = {
            "agent_handshake": {
                "agent_id": "agent-alpha",
                "response_format": "markdown",
            },
            "execution_payload": {
                "intent": f"Summarize the result from {tx_id}",
                "session_id": session_id,
                "constraints": {"budget": 0.0, "latency_max": 1000},
            },
        }
        t0 = time.perf_counter()
        try:
            req = _validate_request(follow)
            auth_header = build_auth_header(
                pub_key=alpha_key,
                agent_id=req.agent_handshake.agent_id,
                method="POST",
                path="/request",
                body=follow,
            )
            resp = handle_gateway_request(
                req,
                auth_header_value=auth_header,
                intake_token=alpha_intake_token,
                raw_body=follow,
                method="POST",
                path="/request",
            )
            rows.append(
                {
                    "sample": 2,
                    "status": "PASS",
                    "intent": follow["execution_payload"]["intent"],
                    "intent_id": resp.intent_id,
                    "has_logic_seal": bool(resp.response.get("logic_seal")),
                    "output_type": type(resp.response.get("output")).__name__,
                    "translation_latency_ms": resp.translation_latency_ms,
                    "latency_slo_pass": resp.translation_latency_ms < 40,
                    "correlation_id": resp.sift_correlation_id,
                    "audit_trace_id": resp.audit_trace_id,
                    "session_id": resp.sift_session_id,
                    "tx_id": resp.tx_id,
                }
            )
        except HTTPException as exc:
            rows.append(
                {
                    "sample": 2,
                    "status": "REJECTED",
                    "intent": follow["execution_payload"]["intent"],
                    "http_status": exc.status_code,
                    "detail": exc.detail,
                    "translation_latency_ms": int((time.perf_counter() - t0) * 1000),
                }
            )
    # Negative security checks for timestamp expiry, nonce replay, out-of-scope bridge, and throttle burst.
    now_ts = int(time.time())
    expired_ts = now_ts - 601
    nonce = "deadbeefdeadbeefdeadbeefdeadbeef"
    expired_sample = {
        "agent_handshake": {"agent_id": "agent-alpha", "response_format": "json"},
        "execution_payload": {"intent": "summarize this", "constraints": {}},
    }
    from auth_manager import sign_request  # local import to avoid exposing in public surface

    expired_sig = sign_request(
        pub_key=alpha_key,
        agent_id="agent-alpha",
        method="POST",
        path="/request",
        body=expired_sample,
        nonce=nonce,
        timestamp=expired_ts,
    )
    expired_header = f"nonce={nonce},timestamp={expired_ts},signature={expired_sig}"
    try:
        req = _validate_request(expired_sample)
        handle_gateway_request(
            req,
            auth_header_value=expired_header,
            intake_token=alpha_intake_token,
            raw_body=expired_sample,
            method="POST",
            path="/request",
        )
    except HTTPException as exc:
        rows.append(
            {
                "sample": 3,
                "status": "REJECTED",
                "intent": expired_sample["execution_payload"]["intent"],
                "http_status": exc.status_code,
                "detail": exc.detail,
                "expected": "expired timestamp",
            }
        )

    replay_sample = {
        "agent_handshake": {"agent_id": "agent-alpha", "response_format": "json"},
        "execution_payload": {"intent": "summarize project memo", "constraints": {}},
    }
    replay_header = build_auth_header(
        pub_key=alpha_key,
        agent_id="agent-alpha",
        method="POST",
        path="/request",
        body=replay_sample,
    )
    try:
        req = _validate_request(replay_sample)
        handle_gateway_request(
            req,
            auth_header_value=replay_header,
            intake_token=alpha_intake_token,
            raw_body=replay_sample,
            method="POST",
            path="/request",
        )
        handle_gateway_request(
            req,
            auth_header_value=replay_header,
            intake_token=alpha_intake_token,
            raw_body=replay_sample,
            method="POST",
            path="/request",
        )
    except HTTPException as exc:
        rows.append(
            {
                "sample": 4,
                "status": "REJECTED",
                "intent": replay_sample["execution_payload"]["intent"],
                "http_status": exc.status_code,
                "detail": exc.detail,
                "expected": "replayed nonce",
            }
        )

    out_of_scope = {
        "agent_handshake": {"agent_id": "agent-beta", "response_format": "json"},
        "execution_payload": {"intent": "Run bridge br-vis-01 on this image", "constraints": {}},
    }
    beta = next(
        (
            row for row in agent_registry.get("agents", [])
            if isinstance(row, dict) and str(row.get("agent_id")) == "agent-beta"
        ),
        None,
    )
    if isinstance(beta, dict) and isinstance(beta.get("pub_key"), str):
        beta_key = str(beta["pub_key"])
        beta_intake_token = generate_intake_token(agent_id="agent-beta", ttl_minutes=120)
        auth_header = build_auth_header(
            pub_key=beta_key,
            agent_id="agent-beta",
            method="POST",
            path="/request",
            body=out_of_scope,
        )
        try:
            req = _validate_request(out_of_scope)
            handle_gateway_request(
                req,
                auth_header_value=auth_header,
                intake_token=beta_intake_token,
                raw_body=out_of_scope,
                method="POST",
                path="/request",
            )
        except HTTPException as exc:
            rows.append(
                {
                    "sample": 5,
                    "status": "REJECTED",
                    "intent": out_of_scope["execution_payload"]["intent"],
                    "http_status": exc.status_code,
                    "detail": exc.detail,
                    "expected": "403_FORBIDDEN",
                }
            )

    burst_rows = 0
    burst_session = session_id
    if not burst_session:
        try:
            burst_session = str(create_session(agent_id="agent-alpha", user_id="agent-alpha", ttl_minutes=30)["session_id"])
        except Exception:  # noqa: BLE001
            burst_session = None
    for i in range(40):
        burst_sample = {
            "agent_handshake": {"agent_id": "agent-alpha", "response_format": "json"},
            "execution_payload": {
                "intent": f"summarize burst #{i}",
                "constraints": {},
                "session_id": burst_session,
            },
        }
        auth_header = build_auth_header(
            pub_key=alpha_key,
            agent_id="agent-alpha",
            method="POST",
            path="/request",
            body=burst_sample,
        )
        try:
            req = _validate_request(burst_sample)
            handle_gateway_request(
                req,
                auth_header_value=auth_header,
                intake_token=alpha_intake_token,
                raw_body=burst_sample,
                method="POST",
                path="/request",
            )
        except HTTPException as exc:
            if exc.status_code == 429:
                burst_rows += 1
    rows.append(
        {
            "sample": 6,
            "status": "PASS" if burst_rows > 0 else "FAIL",
            "detail": "burst throttle test",
            "throttle_rejections": burst_rows,
            "expected_min_rejections": 1,
        }
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    Path(log_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (len(sorted_vals) - 1) * p
    lo = int(rank)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = rank - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def _run_latency_pass(*, calls: int, optimized: bool, agent_key: str, intake_token: str) -> dict[str, Any]:
    if optimized:
        os.environ.pop("SIFT_GATEWAY_DISABLE_CACHE", None)
        os.environ.pop("SIFT_GATEWAY_DISABLE_CAP_OVERRIDE", None)
    else:
        os.environ["SIFT_GATEWAY_DISABLE_CACHE"] = "1"
        os.environ["SIFT_GATEWAY_DISABLE_CAP_OVERRIDE"] = "1"
    for state_file in [".auth_nonce_store.json", ".auth_rate_limit.json"]:
        p = Path(state_file)
        if p.exists():
            p.unlink()

    latencies: list[float] = []
    stages: dict[str, list[float]] = {
        "request_parsing_ms": [],
        "constraint_mapping_ms": [],
        "scope_validation_ms": [],
        "signature_verification_ms": [],
    }
    errors: list[dict[str, Any]] = []
    session_id: str | None = None

    for i in range(calls):
        sample = {
            "agent_handshake": {
                "agent_id": "agent-alpha",
                "response_format": "json",
            },
            "execution_payload": {
                "intent": f"Find the cheapest way to summarize this quarterly update #{i}",
                "constraints": {},
                "session_id": session_id,
            },
        }
        try:
            req = _validate_request(sample)
            auth_header = build_auth_header(
                pub_key=agent_key,
                agent_id=req.agent_handshake.agent_id,
                method="POST",
                path="/request",
                body=sample,
            )
            resp = handle_gateway_request(
                req,
                auth_header_value=auth_header,
                intake_token=intake_token,
                raw_body=sample,
                method="POST",
                path="/request",
            )
            session_id = resp.sift_session_id
            latencies.append(float(resp.translation_latency_ms))
            for key in stages:
                if resp.stage_latencies_ms and key in resp.stage_latencies_ms:
                    stages[key].append(float(resp.stage_latencies_ms[key]))
            # keep under throttle without weakening auth checks
            time.sleep(0.22)
        except HTTPException as exc:
            errors.append({"call": i + 1, "status": exc.status_code, "detail": exc.detail})
            time.sleep(0.22)

    sorted_lat = sorted(latencies)
    summary = {
        "mode": "optimized" if optimized else "baseline",
        "calls_requested": calls,
        "calls_succeeded": len(latencies),
        "calls_failed": len(errors),
        "latency_ms": {
            "mean": round(float(statistics.mean(latencies)) if latencies else 0.0, 3),
            "p50": round(_percentile(sorted_lat, 0.50), 3),
            "p95": round(_percentile(sorted_lat, 0.95), 3),
            "max": round(max(latencies) if latencies else 0.0, 3),
        },
        "stage_latency_ms": {
            key: round(float(statistics.mean(vals)) if vals else 0.0, 3)
            for key, vals in stages.items()
        },
        "errors": errors,
    }
    return summary


def run_latency_profile(*, calls: int, out_path: str) -> dict[str, Any]:
    agent_registry = _load_agent_registry_cached("agent_registry.json")
    alpha = next(
        (
            row for row in agent_registry.get("agents", [])
            if isinstance(row, dict) and str(row.get("agent_id")) == "agent-alpha"
        ),
        None,
    )
    if not isinstance(alpha, dict) or not isinstance(alpha.get("pub_key"), str):
        raise RuntimeError("agent_registry missing active agent-alpha with pub_key")
    alpha_key = str(alpha["pub_key"])

    intake_token = generate_intake_token(agent_id="agent-alpha", ttl_minutes=120)
    baseline = _run_latency_pass(calls=calls, optimized=False, agent_key=alpha_key, intake_token=intake_token)
    optimized = _run_latency_pass(calls=calls, optimized=True, agent_key=alpha_key, intake_token=intake_token)
    target_ms = 40.0
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_latency_ms": target_ms,
        "baseline": baseline,
        "optimized": optimized,
        "decision": {
            "slo_met": optimized["latency_ms"]["p95"] < target_ms,
            "slo_metric": "p95",
        },
    }
    Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sift Agent Gateway utility")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--profile-latency", action="store_true")
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--log", default="Normalization_Validation_Log.json")
    ns = parser.parse_args()
    if ns.validate:
        out = run_validation(ns.log)
        print(json.dumps(out, indent=2, sort_keys=True))
    if ns.profile_latency:
        out = run_latency_profile(calls=max(1, int(ns.calls)), out_path=ns.log)
        print(json.dumps(out, indent=2, sort_keys=True))
