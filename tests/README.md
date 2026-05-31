# Test Suite

These tests lock in the **"honest simulation"** contract that the whole project
depends on. They run with **zero market data** — every case is built from
synthetic OHLCV frames (`tests/conftest.py`), because `data/processed/*.parquet`
is gitignored and absent in CI.

## Run

```bash
pip install -r requirements-ci.txt
pytest                     # all tests
pytest tests/test_backtest.py -v
ruff check tests/          # lint
```

## What each file guards

| File | Guarantee under test |
|------|----------------------|
| `test_backtest.py` | Signal at bar *t* fills at *t+1* open (no same-bar / look-ahead fill); round-trip cost equals the documented fee; long/short PnL signs; position sizing scales PnL; last-bar signal has no effect |
| `test_evaluate.py` | Sharpe / Sortino / Calmar / max-drawdown / profit-factor / win-rate computed correctly on hand-checkable curves |
| `test_validator.py` | Compliance gate rejects `.shift(-n)`, future indexing, bad timeframe, excess leverage, synthetic resample, manual MTF, datetime floor-divide, and un-delayed HTF fractals; the shipped `strategy.py` always passes |
| `test_mtf_data.py` | Higher-timeframe values only become visible **after** the HTF candle closes (the subtlest look-ahead source); `additional_delay_bars` pushes visibility later |
| `test_research_rules.py` | Train/test keep-or-discard thresholds match the documented rules; 0-trade strategies never pass |

## Adding a strategy?

The engine never trusts strategy code — it shifts signals by one bar and the
validator scans for look-ahead. If you add engine behaviour, add a test here
first so the guarantee can't silently regress.
