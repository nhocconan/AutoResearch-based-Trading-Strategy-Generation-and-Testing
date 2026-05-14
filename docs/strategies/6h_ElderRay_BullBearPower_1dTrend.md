# Strategy: 6h_ElderRay_BullBearPower_1dTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.025 | +16.9% | -16.1% | 260 | FAIL |
| ETHUSDT | 0.069 | +21.5% | -15.1% | 262 | PASS |
| SOLUSDT | 1.161 | +240.1% | -28.8% | 235 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.116 | +7.0% | -10.6% | 93 | PASS |
| SOLUSDT | 0.256 | +10.1% | -11.5% | 91 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with 1d EMA34 trend filter identifies strong directional moves in both bull and bear markets. In bull markets, buy when Bull Power > 0 and rising with 1d uptrend; in bear markets, sell when Bear Power > 0 and rising with 1d downtrend. Uses volume confirmation to avoid weak moves.
"""

name = "6h_ElderRay_BullBearPower_1dTrend"
timeframe = "6h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 and rising + 1d uptrend + volume spike
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and close[i] > ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.3:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 and rising + 1d downtrend + volume spike
            elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] and close[i] < ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.3:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 or 1d trend turns down
            if bull_power[i] <= 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power <= 0 or 1d trend turns up
            if bear_power[i] <= 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 22:24
