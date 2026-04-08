# Strategy: mtf_6h_camarilla_vol_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.321 | +1.7% | -19.9% | 340 | DISCARD |
| ETHUSDT | -0.467 | -12.2% | -24.2% | 337 | DISCARD |
| SOLUSDT | 0.397 | +59.2% | -27.8% | 323 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | -0.138 | +1.4% | -16.3% | 100 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #059: 6h Camarilla Pivot + 12h Volume Spike + 1d Trend Filter

This version keeps the original Camarilla/volume/trend concept, but removes
manual stop-loss / take-profit execution from generate_signals().

The backtest engine already enforces:
- signal at bar t -> fill at bar t+1 open
- fees, slippage, and funding

So this strategy now emits only target positions based on completed-bar data.
It does not simulate intrabar stop hits with high/low or assume fills at
close[i], which previously made execution semantics ambiguous.
"""

import numpy as np
import pandas as pd

from mtf_data import align_htf_to_ltf, get_htf_data


name = "mtf_6h_camarilla_vol_trend_v1"
timeframe = "6h"
leverage = 1.0


def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    n = len(close)

    df_12h = get_htf_data(prices, "12h")
    df_1d = get_htf_data(prices, "1d")

    if len(df_12h) >= 20:
        vol_12h = df_12h["volume"].values.astype(np.float64)
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.full(len(df_12h), np.nan)
        vol_ratio_12h[19:] = vol_12h[19:] / vol_ma_20[19:]
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, np.nan)

    if len(df_1d) >= 50:
        close_1d = df_1d["close"].values.astype(np.float64)
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)

    if len(df_1d) > 0:
        day_high = df_1d["high"].values.astype(np.float64)
        day_low = df_1d["low"].values.astype(np.float64)
        day_close = df_1d["close"].values.astype(np.float64)
        day_range = day_high - day_low

        camarilla_r3_1d = day_close + day_range * 1.1 / 4.0
        camarilla_s3_1d = day_close - day_range * 1.1 / 4.0
        camarilla_r4_1d = day_close + day_range * 1.1 / 2.0
        camarilla_s4_1d = day_close - day_range * 1.1 / 2.0

        camarilla_r3 = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
        camarilla_s3 = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
        camarilla_r4 = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
        camarilla_s4 = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    else:
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
        camarilla_r4 = np.full(n, np.nan)
        camarilla_s4 = np.full(n, np.nan)

    camarilla_mid = (camarilla_r3 + camarilla_s4) / 2.0

    signals = np.zeros(n, dtype=np.float64)
    size = 0.25
    position = 0.0

    warmup = 100
    for i in range(warmup, n):
        if (
            np.isnan(camarilla_r3[i])
            or np.isnan(camarilla_s3[i])
            or np.isnan(camarilla_r4[i])
            or np.isnan(camarilla_s4[i])
            or np.isnan(camarilla_mid[i])
            or np.isnan(vol_ratio_12h_aligned[i])
            or np.isnan(ema_50_1d_aligned[i])
        ):
            signals[i] = position
            continue

        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        volume_spike = vol_ratio_12h_aligned[i] > 1.5

        if position > 0.0:
            exit_long = (
                close[i] >= camarilla_r4[i]
                or close[i] <= camarilla_s4[i]
                or price_below_1d_ema
            )
            if exit_long:
                position = 0.0

        elif position < 0.0:
            exit_short = (
                close[i] <= camarilla_s4[i]
                or close[i] >= camarilla_r4[i]
                or price_above_1d_ema
            )
            if exit_short:
                position = 0.0

        if position == 0.0:
            long_condition = (
                (close[i] <= camarilla_s3[i] * 1.001 and price_above_1d_ema)
                or (close[i] > camarilla_mid[i] and volume_spike and price_above_1d_ema)
            )
            short_condition = (
                (close[i] >= camarilla_r3[i] * 0.999 and price_below_1d_ema)
                or (close[i] < camarilla_mid[i] and volume_spike and price_below_1d_ema)
            )

            if long_condition and not short_condition:
                position = size
            elif short_condition and not long_condition:
                position = -size

        signals[i] = position

    return signals
```

## Last Updated
2026-04-07 04:13
