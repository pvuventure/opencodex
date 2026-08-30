# Genspark API-key failover + credit telemetry

## Status

- Phase 1 (DONE): reactive key failover for key-attributable upstream failures
  (429 rate limit, 402 quota, 401, key-attributable 403, Genspark credit-exhaustion
  error bodies). Classifier: `src/providers/key-failover.ts` + `src/providers/genspark.ts`.
  Recovery kinds `key-429` / `key-quota` / `key-auth` across usage log, analytics, GUI.
- Phase 2 (DONE): optional per-key Genspark credit telemetry.
  `src/providers/genspark-credit.ts` — a pool entry may carry a browser-session `cookie`
  used ONLY against `GET /api/payment/get_credit_balance` (never for inference), cached
  with a 45 s TTL. Rotation soft-skips keys with fresh zero balance (two-pass: telemetry
  can only re-order, never shrink, the candidate set).
- Phase 3 (OPEN): `apiKeyPoolStrategy: failover | round-robin | quota-aware` provider
  knob, plus CLI verbs for the management routes below.

## Deferred CLI verbs (owner: phase-3 of this plan)

These management routes exist and are dashboard/GUI-driven today; their `ocx` CLI verbs
are owed by phase 3:

- `PUT /api/providers/keys/cookie` — attach/clear a key's Genspark credit-telemetry
  cookie (write-only; list responses report `hasCookie` presence only).
- `GET /api/providers/keys/credit` — cached per-key credit snapshot
  (`balance` / `checkedAt` / `state` of `healthy | exhausted | unknown`).

Planned shape: `ocx account key cookie --provider genspark --id <keyId> [--clear]`
(reads the cookie from stdin, never argv) and `ocx account key credit --provider genspark`.

## Security invariants (hold across all phases)

- API key = inference auth; cookie = telemetry only. Cookie is never sent to the LLM
  proxy or any non-Genspark-payment host, never logged, never echoed by management APIs.
- Key rotation logs pool-entry ids only, never key material or labels.
- Telemetry failure of any kind degrades to `unknown` and keeps the key eligible —
  a cookie can never disable inference.
