# Sift

**The governance layer between AI agents and the real world.**

Autonomous agents are executing code, calling APIs, sending emails, moving data. Most frameworks ship with zero governance. The agent decides, the action happens, you find out later.

Sift sits between every tool call and execution. It checks policy, issues a cryptographic receipt, and returns ALLOW or BLOCK. Hard block means nothing runs. Fail-closed means if Sift is unreachable, nothing runs either. One kill switch stops all agents cold.

## Why This Matters

DeerFlow, AutoGPT, LangGraph. Powerful frameworks. Zero governance layer. Your agent can write to prod, call external services, send emails at scale, and there is no checkpoint between "decided" and "done."

Sift is that checkpoint.

## Architecture

```
+-----------------------------------------------------------------+
|                         JASON WALKO                             |
|              * Human - Kill Switch Authority *                  |
|          Override - Policy Owner - Final Arbiter                |
+---------------------------+-------------------------------------+
                            |  command & control
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
  +---------------+ +---------------+ +---------------+
  |     ASTRA     | |    GERALT     | |    BREACH     |
  |   Operator    | |    Auditor    | |  Red-Teamer   |
  |               | |               | |               |
  | Executes work | | Reviews logs  | | Finds gaps in |
  | Requests      | | Patches rules | | policy &      |
  | real-world    | | Validates     | | governance    |
  | actions       | | compliance    | | coverage      |
  +-------+-------+ +-------+-------+ +-------+-------+
          |                 |                 |
          |  action request |<----------------+
          |                 |   nightly improvement loop
          v                 |   Breach finds gap ^
  +=======================+ |   Geralt patches ^
  ||  SIFT GOVERNANCE    ||<+   Astra improves
  ||      LAYER          ||
  ||                     ||
  ||  1. Policy check    ||
  ||  2. Sign receipt    ||
  ||  3. ALLOW or BLOCK  ||
  +==========+=============+
             |
     +-------+--------+
     |                 |
     v                 v
  ALLOW             BLOCK
  Signed receipt    Hard stop
  issued            No execution
     |
     v
+-----------------------------------------------------------------+
|                        REAL WORLD                               |
|                                                                 |
|   [ APIs ]         [ Files ]        [ External Services ]       |
+-----------------------------------------------------------------+
```

## Core Properties

- **Fail-closed**: Sift unreachable means nothing executes
- **Cryptographic receipts**: every ALLOW is signed with Ed25519
- **Hard blocks**: DENY means no execution, full stop
- **Kill switch**: one call halts all agent action instantly
- **Policy-driven**: your rules, not framework defaults
- **Audit trail**: every decision logged and signed

## Status

Live. Running on AWS. Alpha access available.

## Contact

jason@walkosystems.com
