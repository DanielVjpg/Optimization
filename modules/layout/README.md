# Layout & Workflow — planned module

**Status: not implemented.** Placeholder fixing the module's shape and inputs.

## What it will do

Reduce non-value-added motion behind the counter. A café bar is a small job
shop: the same few trips (espresso machine → milk fridge → steam wand → hand-off)
repeat hundreds of times a day, so a few feet saved per drink compounds into
hours of recovered capacity per week.

## Inputs

| Input | Source | Status |
|---|---|---|
| Product mix (which drinks, how often) | `Dataset.sales` | already loaded by `core` |
| Station list and coordinates | new optional `stations` table | needs a new schema |
| Drink → station sequence (routing) | new optional `routings` table | needs a new schema |
| Floor area | `BusinessProfile.floor_area_sqft` | already declared |

Product mix — the input that decides which trips actually matter — is the table
the inventory module already runs on.

## Intended approach

1. **From-to chart.** Weight each station pair by trip frequency, taken from
   the observed product mix and drink routings.
2. **Distance matrix.** Rectilinear distance between current station positions.
3. **Score the current layout.** Total travel = Σ (trips × distance), converted
   to minutes per day and labour dollars per year.
4. **Improve it.** The quadratic assignment problem is NP-hard, so use pairwise
   exchange (CRAFT-style) from the current layout — it produces defensible,
   incremental moves an owner can actually make, which matters more here than
   proving optimality.
5. **Report.** Spaghetti diagram before/after, travel reduction, and the moves
   ranked by benefit against how disruptive they are.

## Files when built

```
layout/
├── module.py        LayoutModule(OptimizationModule)
├── flow.py          from-to chart from product mix and routings
├── distance.py      rectilinear distance matrix
├── improve.py       pairwise-exchange layout improvement
└── ui.py            Streamlit rendering, spaghetti diagram
```
