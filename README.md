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

## Contact

jason@walkosystems.com
[sift.walkosystems.com](https://sift.walkosystems.com)
