# Strategy: 4h_TripleConfirmation_Breakout_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.418 | +39.6% | -9.8% | 156 | PASS |
| ETHUSDT | 0.439 | +46.0% | -9.8% | 149 | PASS |
| SOLUSDT | 0.945 | +129.6% | -21.3% | 126 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.004 | -2.9% | -6.0% | 55 | FAIL |
| ETHUSDT | 0.837 | +19.4% | -6.6% | 52 | PASS |
| SOLUSDT | 0.140 | +7.5% | -9.2% | 44 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_TripleConfirmation_Breakout_v1
# Hypothesis: Combine Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter to capture high-probability breakouts in both bull and bear markets.
# Uses strict entry conditions to limit trades (~25-35/year) and avoid fee drag. Exits on mean reversion to middle of channel.
# Designed for low turnover and high edge by requiring confluence of price, volume, and trend.

name = "4h_TripleConfirmation_Breakout_v1"
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
    volume = prices['volume'].values

    # Donchian Channel (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(close[i-20:i])
        lower[i] = np.min(close[i-20:i])
        middle[i] = (upper[i] + lower[i]) / 2

    # Volume confirmation: current volume > 2.0 x 20-period average (stricter to reduce trades)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA50 for trend filter (HTF) - computed once outside loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above upper Donchian band with volume spike and price above 1d EMA50 (uptrend)
            if close[i] > upper[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below lower Donchian band with volume spike and price below 1d EMA50 (downtrend)
            elif close[i] < lower[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below middle of Donchian channel (mean reversion)
            if close[i] < middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above middle of Donchian channel
            if close[i] > middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 05:01
