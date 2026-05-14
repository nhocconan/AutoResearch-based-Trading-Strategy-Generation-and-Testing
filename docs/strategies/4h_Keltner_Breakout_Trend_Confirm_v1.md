# Strategy: 4h_Keltner_Breakout_Trend_Confirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.284 | +6.7% | -11.5% | 158 | FAIL |
| ETHUSDT | 0.611 | +61.0% | -12.0% | 133 | PASS |
| SOLUSDT | 0.624 | +81.3% | -22.2% | 123 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.782 | +18.3% | -6.5% | 45 | PASS |
| SOLUSDT | 0.134 | +7.4% | -12.8% | 45 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Keltner_Breakout_Trend_Confirm_v1
# Hypothesis: Keltner Channel breakouts (ATR-based) combined with 12h EMA trend and volume
# confirmation work in both bull and bear markets. Breakouts capture momentum,
# while the EMA filter ensures we trade with the higher timeframe trend.
# Volume filter reduces false breakouts. Designed for ~30-40 trades/year to minimize fee drag.

name = "4h_Keltner_Breakout_Trend_Confirm_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)

    close_12h = df_12h['close'].values

    # Calculate ATR for Keltner Channels (20-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values

    # Calculate 20-period EMA for Keltner Center
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Keltner Channels: EMA(20) ± 2 * ATR
    keltner_upper = ema20 + 2 * atr
    keltner_lower = ema20 - 2 * atr

    # Get 12h EMA21 for trend filter
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)

    # Volume confirmation: 1.8x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.8

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after indicators need 20 bars
        # Skip if any required data is NaN
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(ema21_12h_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Keltner Upper + volume + 12h uptrend
            if (close[i] > keltner_upper[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema21_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Keltner Lower + volume + 12h downtrend
            elif (close[i] < keltner_lower[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema21_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Keltner Center OR 12h trend turns down
            if close[i] < ema20[i] or close[i] < ema21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Keltner Center OR 12h trend turns up
            if close[i] > ema20[i] or close[i] > ema21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 19:21
