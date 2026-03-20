# Data Pipeline

## Source

Binance Public Data: `https://data.binance.vision/data/futures/um/`

## Symbols

| Symbol | Type | Available From |
|--------|------|---------------|
| BTCUSDT | USDT-M Perpetual | 2020-01 |
| ETHUSDT | USDT-M Perpetual | 2020-01 |
| SOLUSDT | USDT-M Perpetual | 2020-09 |

## Data Types Downloaded

### Klines (OHLCV)
- Timeframes: 1m, 5m, 15m, 1h, 4h, 1d
- Source: `monthly/klines/{SYMBOL}/{INTERVAL}/`
- Columns: open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume

### Funding Rates
- Source: `monthly/fundingRate/{SYMBOL}/`
- Columns: calc_time, funding_interval_hours, last_funding_rate
- Applied every 8 hours (3x daily)

## Storage Format

All data is stored as Parquet files with Snappy compression:

```
data/processed/
├── klines/
│   ├── BTCUSDT/
│   │   ├── 1m.parquet    (~100MB, ~2.6M rows for 2021-2025)
│   │   ├── 5m.parquet    (~20MB)
│   │   ├── 15m.parquet   (~7MB)
│   │   ├── 1h.parquet    (~2MB, ~35K rows)
│   │   ├── 4h.parquet    (~500KB)
│   │   └── 1d.parquet    (~60KB)
│   ├── ETHUSDT/
│   └── SOLUSDT/
└── funding/
    ├── BTCUSDT/
    │   └── funding_rate.parquet
    ├── ETHUSDT/
    └── SOLUSDT/
```

## Download Commands

```bash
# Download everything
python prepare.py

# Download specific symbols/timeframes
python prepare.py --symbols BTCUSDT --timeframes 1h 4h

# Process only (if raw data already downloaded)
python prepare.py --process-only
```

## Data Quality Notes

- Monthly files preferred over daily (fewer downloads, same data)
- SHA256 checksums verified on download
- Duplicates removed during processing
- Timestamps are UTC (timezone-aware)
- No header row in kline CSVs (columns assigned programmatically)
- Funding rate CSVs have header rows

## Last Updated
2026-03-20
