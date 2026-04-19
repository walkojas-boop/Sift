# Sift

**The governance layer between AI agents and the real world.**

Autonomous agents are executing code, calling APIs, sending emails, moving data. Most frameworks ship with zero governance. The agent decides, the action happens, you find out later.

You cannot serialize ambiguity. Execution requires a definitive, unambiguous state. Sift is the resolution layer.

Sift sits between every tool call and execution. It checks policy, issues a cryptographic receipt, and returns ALLOW or BLOCK. Hard block means nothing runs. Fail-closed means if Sift is unreachable, nothing runs either. One kill switch stops all agents cold.

## Why This Matters

DeerFlow shipped without a governance layer. AutoGPT, LangGraph, CrewAI — powerful frameworks, zero checkpoints. Your agent can write to prod, call external services, send emails at scale, and there is no checkpoint between "decided" and "done."

Sift is that checkpoint.

The most capable AI isn't the most deployable AI. The most trustworthy AI is. Capability without governance is liability that hasn't materialized yet.

## How It Works

```
  Agent action request
          |
          v
  +======================+
  ||  SIFT GOVERNANCE  ||
  ||      LAYER        ||
  ||                   ||
  ||  1. Policy check  ||
  ||  2. Sign receipt  ||
  ||  3. ALLOW or BLOCK||
  +=========+===========+
            |
       +----+--------+
       |             |
       v             v
    ALLOW          BLOCK
  Signed receipt   Hard stop
  issued           No execution
       |
       v
  Real world execution
  (APIs, files, external services)
```

## Core Properties

- **Fail-closed**: Sift unreachable means nothing executes
- **Cryptographic receipts**: every ALLOW is signed with Ed25519
- **Hard blocks**: DENY means no execution, full stop
- **Kill switch**: one call halts all agent action instantly
- **Policy-driven**: your rules, not framework defaults
- **Audit trail**: every decision logged and signed
- **Runtime enforcement**: policy checks happen before the action fires, not after

## What This Solves

Most agent safety is probabilistic — system prompts, guidelines, asking the agent to "be careful." That works most of the time.

Sift is deterministic. The policy layer enforces at runtime. The agent cannot bypass it. That is the difference between hoping it stays in bounds and knowing it does.

## Status

Live. Running on AWS. Real tenants. Real receipts.

## Verify It Yourself

Sift isn't a slide deck — it's a live API that returns cryptographically-signed receipts. Two sample receipts from production, captured on 2026-04-19, are in a public gist with the exact Python script that produced them:

**[End-to-end proof gist →](https://gist.github.com/walkojas-boop/8cd9962db93496aa87b1a3ec0a86fb9f)**

Includes:

- `sift_end_to_end_proof.py` — pure stdlib + `cryptography`, runs the full `/auth/challenge` → sign → `/authorize` → verify chain against `https://sift.walkosystems.com/api/v1/`
- `allow_receipt.json` — real ALLOW from production, receipt `f9db7cef-09c3-4e25-b722-881c6af3bbbd`, issued by Sift `v4.1.0` on ECS task revision 12
- `deny_receipt.json` — real DENY (`ROLE_NOT_CONFIGURED`), receipt `b6a9700a-2a66-431f-bb1a-542a41f59b7c`. Same tenant, same agent, same policy bundle — just a role that isn't configured

Both receipts are ed25519-signed by the same gateway key (`receipt-k1-20260218-214620`) and pinned to the same container image digest (`sha256:21180bca554de634…`). The signature + `canonical_payload` in each receipt means a third party can independently verify what Sift decided — no trust in the transport required.

Same request, different role, different decision, both signed. That's the product.

## Contact

jason@walkosystems.com
[sift.walkosystems.com](https://sift.walkosystems.com)


## Agent Discovery Protocol

Sift is part of a 14-service agent infrastructure stack, all discoverable via one endpoint:

```
GET https://walkosystems.com/.well-known/agent-discovery.json
```

See the [Agent Discovery Protocol spec](https://github.com/walkojas-boop/agent-discovery-protocol) for details.
