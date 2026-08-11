# Staffing & Scheduling — planned module

**Status: not implemented.** This folder exists to fix the shape of the module
before it is built, and to prove the architecture holds: everything below is
reachable from the same `Dataset` the inventory module already consumes.

## What it will do

Turn forecast demand into a staffing plan: how many people, at which stations,
in which half-hour blocks, and then a roster that covers it at minimum cost.

## Inputs

| Input | Source | Status |
|---|---|---|
| Transaction history (date, item, quantity) | `Dataset.sales` | already loaded by `core` |
| Transaction timestamps (intra-day) | new optional column on the sales schema | needs `core/schemas.py` extension |
| Labour rate, service-time assumptions | `BusinessProfile.labor_rate_per_hour`, `avg_service_time_seconds` | already declared |
| Staff availability / shift rules | new optional `staff` table | needs a new schema |

Note the pattern: two of four already exist, and the two that do not are
additive changes to `core` — no change to the inventory module.

## Intended approach

1. **Demand → arrivals.** Reuse `modules.inventory.forecasting` to forecast
   daily volume, then apply an intra-day profile to get arrivals per half-hour.
   The forecasting layer is item-agnostic, so this is reuse, not a rewrite.
2. **Arrivals → required servers.** Size each interval with an M/M/c queueing
   model against a target wait (e.g. 90% of customers served within 3 minutes).
   The square-root staffing rule gives the first cut.
3. **Required servers → roster.** Set-covering / integer program over feasible
   shifts, minimising labour cost subject to coverage, minimum shift length,
   break rules and staff availability.
4. **Report.** Cost of the current schedule vs. the optimised one, plus the
   intervals that are chronically over- or under-covered.

## Files when built

```
staffing/
├── module.py        StaffingModule(OptimizationModule)
├── arrivals.py      daily forecast -> intra-day arrival profile
├── queueing.py      M/M/c sizing, square-root staffing rule
├── scheduling.py    shift set-covering model
└── ui.py            Streamlit rendering
```
