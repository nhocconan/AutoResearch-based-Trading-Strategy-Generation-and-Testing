# Strategy: 6h_VWAP_Trend_Follower

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.083 | +17.5% | -8.5% | 80 | FAIL |
| ETHUSDT | 0.439 | +45.1% | -9.5% | 87 | PASS |
| SOLUSDT | -0.354 | -7.9% | -29.6% | 75 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.301 | +9.4% | -5.9% | 27 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_VWAP_Trend_Follower
Hypothesis: Use VWAP deviation from 6h price with 12h trend filter and volume confirmation to capture trends. 
VWAP acts as dynamic support/resistance; price reverting to VWAP in trending markets offers high-probability entries.
Works in bull/bear markets by filtering with 12h trend and requiring volume spikes to avoid chop. 
Target: 15-30 trades/year per symbol.
"""

name = "6h_VWAP_Trend_Follower"
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

    # Get 12h data for trend filter (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate VWAP for 6h (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    # Avoid division by zero at start
    vwap = np.where(vwap_den == 0, typical_price, vwap)

    # VWAP deviation as percentage
    vwap_dev = (close - vwap) / vwap * 100.0

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        vwap_dev_val = vwap_dev[i]
        ema50_val = ema50_12h_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(vwap_dev_val) or np.isnan(ema50_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price below VWAP (oversold) + 12h uptrend + volume spike
            if vwap_dev_val < -0.5 and close[i] > ema50_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price above VWAP (overbought) + 12h downtrend + volume spike
            elif vwap_dev_val > 0.5 and close[i] < ema50_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above VWAP or 12h trend turns down
            if vwap_dev_val > 0.0 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below VWAP or 12h trend turns up
            if vwap_dev_val < 0.0 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 21:56
