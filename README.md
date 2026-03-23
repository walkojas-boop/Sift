# Sift

**The governance layer between AI agents and the real world.**

Autonomous agents are executing code, calling APIs, sending emails, moving data. Most frameworks ship with zero governance. The agent decides, the action happens, you find out later.

Sift sits between every tool call and execution. It checks policy, issues a cryptographic receipt, and returns ALLOW or BLOCK. Hard block means nothing runs. Fail-closed means if Sift is unreachable, nothing runs either. One kill switch stops all agents cold.

## Why This Matters

DeerFlow, AutoGPT, LangGraph. Powerful frameworks. Zero governance layer. Your agent can write to prod, call external services, send emails at scale, and there is no checkpoint between "decided" and "done."

Sift is that checkpoint.

## Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                         JASON WALKO                             â”‚
â”‚                  â—† Human Â· Kill Switch Authority â—†              â”‚
â”‚            Override Â· Policy Owner Â· Final Arbiter              â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚  command & control
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
          â”‚                 â”‚                 â”‚
          â–¼                 â–¼                 â–¼
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚     ASTRA     â”‚ â”‚    GERALT     â”‚ â”‚    BREACH     â”‚
  â”‚   Operator    â”‚ â”‚    Auditor    â”‚ â”‚  Red-Teamer   â”‚
  â”‚               â”‚ â”‚               â”‚ â”‚               â”‚
  â”‚ Executes work â”‚ â”‚ Reviews logs  â”‚ â”‚ Finds gaps in â”‚
  â”‚ Requests      â”‚ â”‚ Patches rules â”‚ â”‚ policy &      â”‚
  â”‚ real-world    â”‚ â”‚ Validates     â”‚ â”‚ governance    â”‚
  â”‚ actions       â”‚ â”‚ compliance    â”‚ â”‚ coverage      â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
          â”‚                 â”‚                 â”‚
          â”‚  action request â”‚â—„â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
          â”‚                 â”‚   nightly improvement loop
          â–¼                 â”‚   Breach finds gap â†’
  â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•— â”‚   Geralt patches â†’
  â•‘   SIFT GOVERNANCE     â•‘â—„â”˜   Astra improves
  â•‘       LAYER           â•‘
  â•‘                       â•‘
  â•‘  1. Policy check      â•‘
  â•‘  2. Sign receipt      â•‘
  â•‘  3. ALLOW or BLOCK    â•‘
  â•šâ•â•â•â•â•â•â•â•â•â•â•¤â•â•â•â•â•â•â•â•â•â•â•â•â•
             â”‚
     â”Œâ”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”
     â”‚                â”‚
     â–¼                â–¼
  âœ… ALLOW         âŒ BLOCK
  Signed receipt   Hard stop
  issued           No execution
     â”‚
     â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        REAL WORLD                               â”‚
â”‚                                                                 â”‚
â”‚   [ APIs ]         [ Files ]        [ External Services ]       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
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
