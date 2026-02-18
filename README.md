# Sift

https://tungsten-holly-695.notion.site/Sift-Infrastructure-Overview-f6a7a2c301d544ac9a3973dd2c6caa22

Sift is an agent execution routing and governance layer.

It sits between agent intent and tool/model execution.

Core functions:

• Policy gating before execution  
• Tool permission enforcement  
• Latency / cost / certainty routing  
• Retry + fallback orchestration  
• Signed telemetry + audit receipts  

Sift exposes an intake handshake endpoint for agent onboarding and routing validation.

---

## Intake Harness

POST /request

Agent handshake supported.

Returns:

• Routing decision trace  
• Selected bridge/tool  
• Fallback + retry log  
• Signed telemetry receipt  

---

## Test Harness

Example execution payload:

"Translate this sentence to Spanish: We are ready for alpha validation."

Used to validate:

• Intent resolution  
• Bridge selection  
• Governance enforcement  
• Telemetry emission  

---

Looking to connect with builders working on:

• Agent infrastructure  
• Tool permissioning  
• Policy enforcement  
• Execution routing  
• Telemetry + auditability

Sift Alpha Gateway — Security & Governance Validation Summary
Overview

The Sift Alpha Gateway has completed a full adversarial validation cycle across execution, identity, governance, and control-plane layers. The system now operates as a deterministic, fail-closed execution governor — not a permissive router.

All tests were conducted under mock and live alpha conditions.

Security Layers Validated
1. Temporal Gating
Blocks stale requests (>300s old)
Blocks future-dated requests (>5s ahead)
Fail-closed on clock or validation failure
Fully logged with structured telemetry

2. Replay Protection
Atomic replay cache (120s TTL)
Verified under concurrent burst (20 simultaneous identical requests)
Result: 1 permit / 19 deterministic denials
Fail-closed on cache failure

3. Scope & Risk Enforcement
Agent → action → tool ACL binding
Max risk-tier enforcement per agent
Unauthorized tool access blocked pre-routing
No routing engine side-effects during denial

4. Parameter Integrity
Strict tool manifest parameter whitelisting
Injection attempts rejected (422 schema violations)
No capability override via param pollution

5. Governance Saturation Resilience
Sustained denial flood (~50 req/sec)
100% malicious rejection accuracy
100% valid request success
No fail-open behavior
No telemetry drops

6. Cryptographic Identity Model (ED25519)
Public-key based identity (no shared secrets)
Challenge → Sign → Verify handshake
Full envelope binding:
request_id
tenant_id
agent_id
action
tool
risk_tier
nonce
timestamp
params_hash
Any payload mutation invalidates signature

7. Instant Revocation (Kill-Switch)
Agent revocation effective immediately
No restart dependency
Old receipts invalidated
Clear telemetry separation:
identity_denied
receipt_denied_revoked

8. Control Plane Integrity
policy_hash + policy_version attached to each decision
Manifest change detection with integrity alerts
Boot-time audit of active security thresholds
Security thresholds exposed via /metrics
No silent privilege drift
Architectural Properties Achieved
Deterministic policy enforcement
Strict fail-closed behavior
Cryptographically anchored identity
Air-gapped routing (no execution on denial)
Versioned, hashed governance layer
High-fidelity structured telemetry

Alpha Status
The Sift Alpha Gateway has reached Operational Stability under single-node deployment conditions.
It is hardened against
Replay attacks
Stale/future timestamp attacks
Privilege escalation
Parameter injection
Resource exhaustion
Key compromise
Manifest tampering
Known Future Scaling Considerations
Centralized replay store for horizontal scaling
Distributed identity cache
Formalized key rotation workflows
External audit preparation
Sift now governs actions deterministically not intentions through cryptographic identity anchoring and policy-bound execution routing.
Deployment Architecture

Sift is deployed as a fully cloud-native service on AWS.

  Infrastructure Stack

  Amazon ECS (Fargate) — container orchestration

  Application Load Balancer (ALB) — public ingress layer

  Target Group (IP mode) — Fargate-compatible routing
 
  AWS ECR — container image registry
  
  CloudWatch Logs — centralized logging
  
  VPC + Security Groups — least-privilege network isolation
  
  Runtime Configuration
  
  FastAPI application containerized via Docker
  
  Uvicorn running on 0.0.0.0:8080
  
  Health check endpoint: /healthz
  
  ALB listener: HTTP :80
  
  Target group forwards to container port 8080
  
  ECS service registers tasks dynamically via awsvpc networking

Network Security Model

  ALB security group:

  Inbound: TCP 80 from 0.0.0.0/0

Service security group:

  Inbound: TCP 8080 from ALB security group only

  No direct public access to tasks

Traffic flows:

  Client → ALB → Target Group → Fargate Task

  Deployment Properties

  Rolling deployments enabled

  Circuit breaker enabled

  Health-check-based registration/deregistration

  Fail-closed behavior enforced at application layer
