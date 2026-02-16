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

Sift Lite

Real Routing Examples

Results
Parsed: use_case=crypto trading, budget=$200, time=5days, experience=advanced
1. OpenAI Assistants
Platform | Fit: 100
Why: good use-case match; fits your budget; easy setup for your current experience
Label: standard
2. CCXT
Trading/Exchange Execution | Fit: 100
Why: strong use-case match; near your budget range; moderate setup effort; stretch recommendation
Label: stretch
3. Hummingbot
Trading/Exchange Execution | Fit: 100
Why: strong use-case match; near your budget range; moderate setup effort; stretch recommendation
Label: stretch

Results
Parsed: use_case=customer support, budget=$100, time=2h, experience=intermediate
1. OpenAI Assistants
Platform | Fit: 100
Why: good use-case match; fits your budget; easy setup for your current experience
Label: standard
2. Zapier
Automation Tool | Fit: 95.35
Why: good use-case match; fits your budget; easy setup for your current experience
Label: standard
3. Make
Automation Tool | Fit: 66
Why: partial use-case match; fits your budget; easy setup for your current experience
Label: standard

Results
Parsed: use_case=I run a law firm and need to automate legal citations and filings. I am a normie but I have capital, budget=$100, time=2h, experience=intermediate
1. Zapier
Automation Tool | Fit: 66
Why: partial use-case match; fits your budget; easy setup for your current experience
Label: standard
2. Make
Automation Tool | Fit: 66
Why: partial use-case match; fits your budget; easy setup for your current experience
Label: standard
3. Pipedream
Automation Tool | Fit: 41
Why: partial use-case match; fits your budget; setup may require more time or experience; stretch recommendation
Label: stretch

Results
Parsed: use_case=customer support, budget=$100, time=2h, experience=beginner
1. Zapier
Automation Tool | Fit: 100
Why: good use-case match; fits your budget; easy setup for beginners
Label: standard
2. OpenAI Assistants
Platform | Fit: 100
Why: good use-case match; fits your budget; easy setup for beginners
Label: standard
3. Make
Automation Tool | Fit: 89.1
Why: partial use-case match; fits your budget; easy setup for beginners
Label: standard

Results
Parsed: use_case=email automation, budget=$100, time=2h, experience=intermediate
1. Zapier
Automation Tool | Fit: 100
Why: strong use-case match; fits your budget; easy setup for your current experience
Label: standard
2. Make
Automation Tool | Fit: 80.65
Why: partial use-case match; fits your budget; easy setup for your current experience
Label: standard
3. Pipedream
Automation Tool | Fit: 63.89
Why: good use-case match; fits your budget; setup may require more time or experience; stretch recommendation
Label: stretch
