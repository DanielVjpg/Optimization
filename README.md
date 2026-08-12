# Optimization

Operations analysis for small businesses — industrial engineering and
operations research applied to the decisions a café or restaurant actually
makes every week.

Most small businesses order by feel. They know roughly how much milk they go
through, they order when the walk-in looks empty, and the cost of getting it
wrong shows up as spoiled product, cash sitting on a shelf, or a Saturday
morning with no oat milk. The methods for doing this properly — demand
forecasting, safety stock, economic order quantity — are seventy years old and
well understood. They are just packaged for businesses with an ERP system and
an analyst, not for a shop with eleven items and a spreadsheet.

**Current scope: the inventory / demand-forecasting module.** Staffing and
layout modules are planned; the architecture is already built to accept them.

---

## What it does today

Given a history of what a shop used and what it bought, the tool:

1. **Forecasts demand** for each item over the next 1–4 weeks, choosing between
   three models per item based on measured out-of-sample accuracy.
2. **Recommends a replenishment policy** — a reorder point and an order
   quantity per item — from safety stock and EOQ, constrained by shelf life,
   case packs and supplier minimums.
3. **Flags over- and under-ordered items** and estimates what the current
   ordering behaviour costs per year, broken down by source.

On the bundled sample data (a synthetic 11-item coffee shop, 120 days) it finds
roughly **$10,600 a year** of opportunity, dominated by three real findings:
oat milk being bought below usage, croissants over-ordered against a two-day
shelf life, and vanilla syrup surplus tying up cash.

---

## Quick start

```bash
git clone <this repo>
cd Optimization

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

The app opens with the sample coffee shop loaded, so it works before you have
any data of your own. Switch the sidebar to **Upload my own** to run it on real
files.

```bash
python -m pytest tests -q          # run the test suite
python data/generate_sample_data.py  # regenerate the sample data
```

---

## Input data

Everything is CSV. Column names are matched loosely — `Date`, `order_date`,
`Item Name`, `Qty`, `Unit Price` and similar variants are recognised — so a POS
or supplier export usually loads without editing. Unrecognised columns are
ignored, and every accepted alias is listed in `core/schemas.py` and shown in
the app under *What data does this need?*.

### `sales` — usage history (required)

| column | required | meaning |
|---|---|---|
| `date` | yes | date the item was used or sold |
| `item` | yes | item name; matched against the catalog |
| `quantity` | yes | quantity consumed, in the item's stock unit |
| `revenue` | no | not used by the inventory module |

### `purchases` — order history (optional but recommended)

| column | required | meaning |
|---|---|---|
| `date` | yes | date ordered or received |
| `item` | yes | item name |
| `quantity` | yes | quantity ordered |
| `unit_cost` | no | overrides the catalog cost |
| `extended_cost` | no | used to derive `unit_cost` when absent |

Without this table the tool still recommends a policy, but it cannot tell you
whether your current ordering is wrong — that comparison is the whole point, so
supply it if you can. If you have *only* purchase data, it is used as a proxy
for demand, with a warning: ordering is lumpier than consumption, so it
overstates variability and inflates safety stock.

### `items` — catalog (optional)

`item` is the only required column. Everything else falls back to a business
profile default, and the app reports which values were inferred.

`unit`, `unit_cost`, `lead_time_days`, `shelf_life_days`, `case_pack`,
`min_order_qty`, `margin_per_unit` (or `sell_price`), `on_hand`, `category`,
`supplier`.

`margin_per_unit` is the contribution margin lost per unit short. For an
ingredient that means the margin of the drinks one stock unit produces — a
gallon of milk making ~22 lattes at ~$2.80 margin is about $62. It drives the
stockout estimate, so it is worth getting roughly right.

### Preparing a real POS export

POS exports are transaction-level, span several locations, and arrive in
whatever delimiter the vendor chose. `data/prepare_pos_export.py` reduces one
to the `sales` table above:

```bash
python data/prepare_pos_export.py raw_export.csv --catalog
python data/prepare_pos_export.py raw_export.csv --store "Astoria" --group-by product_type
```

It detects the delimiter, keeps the busiest location (or `--store`), sums
quantity per item per day, and optionally writes a starter item catalog with
menu prices filled in and cost/lead-time/shelf-life columns left blank.

Two things it handles that matter more than they sound. The delimiter is found
by testing `, | ; \t` and requiring a consistent split, **not** by
`csv.Sniffer` — on the first real export this project met, the sniffer chose
`t`, the letter, because it recurs inside "transaction". And item names are
whitespace-normalised: an export carrying both `"Scottish Cream Scone "` and
`"Scottish Cream Scone"` otherwise splits one product's history in two,
halving its forecast and its reorder point.

Output goes to `data/prepared/`, which is gitignored — it is client data and
is reproducible from the raw export.

### Business profile

Cost and service assumptions are editable in the sidebar, but for consulting
work keep one YAML file per client instead — see
`data/sample_business_profile.yaml`:

```python
from core.business_profile import BusinessProfile
profile = BusinessProfile.from_yaml("clients/riverside_cafe.yaml")
```

Every value in it is an assumption someone can disagree with, so it belongs in
a file you can hand over and argue about rather than buried in the code.
Unknown keys are preserved, so a profile can carry parameters for modules that
do not exist yet.

---

## Method

### Forecasting

Three models compete for every item, behind one interface in
`modules/inventory/forecasting.py`:

| model | what it assumes |
|---|---|
| **Seasonal naive** | tomorrow looks like the same weekday last week — the benchmark |
| **Moving average + weekday** | a flat level from a 28-day trailing average, re-seasonalised |
| **Damped Holt + weekday** | level plus a damped trend, re-seasonalised — the usual winner |

Day-of-week seasonality is not optional here. Coffee-shop demand swings 30–40%
between a Monday and a Saturday, and a model that ignores it under-orders every
weekend and over-orders every Monday. Factors are estimated classically: divide
each day by a centred 7-day moving average to strip level and trend, then take
the **median** ratio per weekday, so one catering order does not redefine
"Tuesday".

The Holt recursions, on the deseasonalised series `x`:

```
forecast_t = level + φ·trend
level_t    = α·x_t + (1-α)·forecast_t
trend_t    = β·(level_t - level_{t-1}) + (1-β)·φ·trend_{t-1}
```

and `h` days ahead: `(level + trend·Σφⁱ) × weekday_factor`.

`φ = 0.95` damps the trend. Undamped Holt extrapolated four weeks out will
cheerfully predict a café sells 40% more milk than last month because of a
fortnight of good weather. `α` and `β` are grid-searched per item rather than
fixed, because a busy staple and a slow syrup want different smoothing.

**Model selection is by measured accuracy, not by taste.** Each model is scored
by rolling-origin backtesting — fit on the past, predict an unseen block,
repeat at four cut points — and the lowest-MAE model wins *for that item*. In-
sample fit is reported nowhere, because it rewards overfitting. Item-level
selection matters: milk is a smooth trending series, matcha is near-flat with
sporadic zeros, and one model does not serve both.

### Safety stock

```
SS = z · √(L·σ² + d²·σ_L²)
```

`z` comes from the target service level (95% → 1.645). The first term is demand
uncertainty over the lead time `L`; the second covers unreliable delivery and
drops out when `σ_L = 0`, reducing to the familiar `z·σ·√L`.

**`σ` is forecast error, not demand variability**, measured out-of-sample from
the backtest residuals. This is the one place the design departs from the
textbook version, and it matters: predictable weekly swings are already carried
in the reorder point, so charging for them again in buffer stock means paying
twice for the same variation. On the sample data, forecast-error σ runs about
half of raw demand σ — that difference is inventory a shop does not have to
hold.

### Reorder point

```
ROP = forecast demand over the lead time + safety stock
```

Lead-time demand comes from the **dated forecast**, not an average, so an order
placed Thursday for Saturday delivery is sized for a weekend.

### Order quantity

```
EOQ = √(2DS/H)        TC(Q) = (D/Q)·S + (Q/2)·H
```

`D` is annualised from the forward forecast, `H = unit_cost × holding_rate`,
`S` is the cost of ordering one item line. EOQ is then constrained, in order:

1. **Shelf life** — an order must be consumed within `shelf_life_utilization`
   (default 0.5) of the item's usable life. Nobody wants to serve the last
   gallon on its expiry date.
2. **Maximum days of cover** (default 30) — a stand-in for the shelf space and
   cash a small business actually has.
3. **Supplier minimum**, then **round up to a whole case**.

The constraint that actually bound is reported per item, and a conflict — a
case pack larger than the shelf life supports — is called out explicitly. On
the sample data that fires for croissants: the smallest orderable tray is more
than a day's demand of a two-day product, which is a supplier conversation, not
a forecasting problem.

> **On `S`, the order cost.** It is the *incremental* cost of adding one item
> line to a delivery — counting it, receiving it, putting it away — defaulting
> to $4, not the cost of a whole truck. A café orders thirty items from one
> distributor at once; charging each item the full delivery cost inflates every
> EOQ into months of stock. Handling the shared cost properly is the joint
> replenishment problem, and it is the single highest-value v2 upgrade in this
> module. The `max_days_of_cover` cap is the honest interim backstop.

### Costing the inefficiency

Two comparisons: ordered vs. used over the history window, and current lot size
vs. EOQ. Four sources, each annualised and each recording its assumptions:

| source | basis |
|---|---|
| Spoilage | perishable surplus beyond one shelf life of demand, written off at cost |
| Carrying cost | surplus value × holding rate |
| Lot sizing | `TC(Q_current) − TC(Q_recommended)` |
| Stockouts | projected unmet demand × `stockout_realization_rate` × margin |

Replenishment is a sawtooth, so purchases and usage never balance exactly over
a finite window — where the last delivery landed leaves a few percent either
way. Nothing is priced until the imbalance clears a threshold (10% over, 8%
under) *and* a materiality floor, because recommending action on a $3
discrepancy costs more attention than it returns.

The stockout figure assumes only half of unmet demand is truly lost — when the
oat milk runs out, some customers take whole milk and some walk. That default
is adjustable and stated on the finding, because it is the assumption a
skeptical owner will challenge first.

**Honest limitation:** the four figures are added but overlap slightly. Carrying
cost is charged on accumulated surplus while lot-sizing cost is charged on
average cycle stock, and those pools are not perfectly disjoint. Treat the total
as an order of magnitude and the per-item findings as the actionable part. A
number an owner can argue with is worth more than a big number they cannot
believe.

---

## Architecture

```
Optimization/
├── app/
│   └── streamlit_app.py        # shell: data in, module picker, dispatch
├── core/                       # module-agnostic foundation
│   ├── schemas.py              # canonical tables + alias mapping
│   ├── data_loader.py          # the ONLY place files are read
│   ├── dataset.py              # Dataset: the shared input object
│   ├── business_profile.py     # operating & cost assumptions
│   ├── module_base.py          # OptimizationModule ABC — the contract
│   ├── registry.py             # module discovery
│   ├── metrics.py              # MAE / RMSE / MAPE / bias
│   └── viz.py                  # shared Altair charts
├── modules/
│   ├── inventory/              # the only built module
│   │   ├── forecasting.py      # models, backtesting, selection
│   │   ├── policy.py           # safety stock, ROP, EOQ + constraints
│   │   ├── diagnostics.py      # over/under-ordering, cost estimates
│   │   ├── module.py           # orchestration
│   │   └── ui.py               # Streamlit rendering
│   ├── staffing/               # PLANNED — README + stub only
│   └── layout/                 # PLANNED — README + stub only
├── data/
│   ├── sample/                 # synthetic coffee-shop CSVs
│   └── generate_sample_data.py
└── tests/
```

Three rules keep this extensible:

**`core` knows about businesses; modules know about problems.** Anything true of
a business regardless of what you are optimising — how to read a CSV, what a
lead time is, what the labour rate is — lives in `core`. Anything only an
inventory analysis cares about lives in `modules/inventory/`.

**Ingestion happens exactly once.** Modules receive a `Dataset` and never touch
files. A staffing module reuses the same loading, cleaning, validation and
alias-mapping code for free, and gains its own inputs by *adding* to the shared
schemas rather than forking them.

**The app renders whatever the registry holds.** `app/streamlit_app.py` never
imports a module directly. Shipping a new module is a folder plus one
`registry.register(...)` line — no edits to the shell, to `core`, or to any
existing module. Planned modules are registered too, so the roadmap is visible
in the product, greyed out in the sidebar.

### Adding a module

```python
class StaffingModule(OptimizationModule):
    key = "staffing"
    display_name = "Staffing & Scheduling"
    required_inputs = ("sales",)

    def run(self, dataset: Dataset, **options) -> ModuleResult: ...
    def render(self, result: ModuleResult) -> None: ...
```

Register it in `modules/__init__.py`, replacing its `register_planned` line.
That is the entire integration. `run` must not call Streamlit — keeping the
analysis headless is what lets it be driven from a test, a notebook, or a
consulting script that exports a client deliverable.

---

## Roadmap

### Inventory — v2 upgrades

Ordered by value, not by novelty:

1. **Joint replenishment.** Items from one supplier share a delivery, so their
   order cycles should be solved together rather than item by item. This is the
   biggest single correction to the current model, and it is why order
   quantities currently need a days-of-cover backstop.
2. **Prophet or SARIMA.** With a year or more of history there is annual
   seasonality and there are holiday effects to capture, and Prophet handles
   both with little tuning. It is not worth it at 120 days of history and a
   two-week horizon — weekly seasonality is nearly all the signal there is, and
   the current models capture it — but it becomes the right call as data
   accumulates.
3. **Exogenous regressors.** Weather, local events, school terms and promotions
   drive a real share of café demand and none of it is in the sales history. A
   gradient-boosted model over these features would beat any purely
   autoregressive approach.
4. **Perishable-specific policy.** The newsvendor model, not EOQ, is the right
   frame for items that expire in days — croissants and pastries especially.
5. **Intermittent demand.** Croston's method for slow movers where zero-demand
   days dominate.
6. **Live tracking.** Compare recommendation to what was actually ordered, and
   measure realised savings rather than projected ones.

### Future modules

**Staffing & Scheduling** — forecast demand → arrivals per half hour → required
staff by M/M/c queueing against a target wait → a roster from a set-covering
integer program. Reuses this module's forecasting layer directly. See
`modules/staffing/README.md`.

**Layout & Workflow** — a from-to chart of barista movement weighted by actual
product mix, scored against a rectilinear distance matrix, improved by pairwise
exchange. The product mix that decides which trips matter is the table the
inventory module already runs on. See `modules/layout/README.md`.

Both placeholder folders document their intended inputs and note which already
exist in `core` — for staffing, two of four do.

---

## Sample data

`data/sample/` holds a synthetic 11-item coffee shop over 120 days, generated
deterministically by `data/generate_sample_data.py`. It is built to exercise the
analysis rather than to be bland: real weekly seasonality (weekends ~35% above
Mondays), a mild growth trend, a holiday week with a closed day, scattered
bad-weather dips, and **three deliberately seeded inefficiencies** — vanilla
syrup over-ordered ~60%, oat milk under-ordered ~22%, and croissants
over-ordered ~20% against a two-day shelf life. The tests assert the tool finds
all three.

It is synthetic, which means it is kinder than reality: no mis-keyed item
names, no missing weeks, no unit changes mid-history. The loader is built for
those; the sample data does not prove it handles them.

---

## Status

Built and tested: the inventory module, end to end, with 150 tests covering the
policy math against hand calculations, forecaster behaviour, ingestion and
cleaning, POS export preparation, and the full pipeline on the sample data.

Run once against a real 149k-row café export (3 stores, 181 days, 80 menu
items). It loads and produces a policy for every item, but accuracy splits
sharply by volume: items selling 10–30/day forecast at ~51% MAE and the
best at ~31%, while the 22 items selling under 1/day — zero on ~124 of 181
days — come out above 175%. That is intermittent demand, and it is the
evidence for moving Croston's method up the v2 list. Aggregating to coarser
product groups does not help; filtering to items above roughly 3/day does.

Not built: staffing and layout — structure and documented intent only.
