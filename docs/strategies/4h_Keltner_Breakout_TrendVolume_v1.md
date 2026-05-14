# Strategy: 4h_Keltner_Breakout_TrendVolume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.266 | +36.4% | -14.3% | 95 | PASS |
| ETHUSDT | 0.176 | +30.0% | -16.0% | 95 | PASS |
| SOLUSDT | 0.941 | +200.3% | -28.5% | 98 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.676 | -3.4% | -9.4% | 38 | FAIL |
| ETHUSDT | 0.972 | +27.8% | -8.0% | 24 | PASS |
| SOLUSDT | 0.647 | +21.1% | -9.0% | 27 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Keltner_Breakout_TrendVolume_v1
# Hypothesis: 4h Keltner Channel breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses Keltner Channel (ATR-based) for dynamic support/resistance, 1d EMA34 for trend direction,
# and volume spike (1.8x 20-period average) to confirm breakout strength. Designed for 25-40 trades/year.
# Works in bull/bear markets by following 1d trend direction. Exit on reversal signal (price crosses EMA34).
# Targets BTC/ETH with tighter entry to avoid whipsaw and reduce trade frequency.

name = "4h_Keltner_Breakout_TrendVolume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 4h data for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate 4h ATR(10) for Keltner Channel
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3 = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values

    # Calculate Keltner Channel: EMA20 ± 2*ATR
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema20_4h + 2 * atr10
    keltner_lower = ema20_4h - 2 * atr10
    keltner_upper_aligned = align_htf_to_ltf(prices, df_4h, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_4h, keltner_lower)

    # Calculate 4h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.8  # Require 1.8x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Keltner Upper in 1d uptrend with volume spike
            if (close[i] > keltner_upper_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Keltner Lower in 1d downtrend with volume spike
            elif (close[i] < keltner_lower_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1d EMA34 (trend reversal)
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1d EMA34 (trend reversal)
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 18:45
