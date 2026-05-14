# Strategy: 4h_RSI_Trend_Momentum_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.084 | +23.9% | -11.1% | 204 | PASS |
| ETHUSDT | -0.316 | +4.2% | -10.2% | 192 | FAIL |
| SOLUSDT | 0.430 | +55.1% | -17.2% | 188 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.026 | +6.2% | -5.1% | 62 | PASS |
| SOLUSDT | 0.363 | +11.0% | -10.1% | 57 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_RSI_Trend_Momentum_Volume
# Hypothesis: Use RSI(14) extremes with 1d EMA50 trend filter and volume confirmation to capture momentum bursts in both bull and bear markets. 
# Long when RSI > 70 in uptrend with volume spike, short when RSI < 30 in downtrend with volume spike. Exit on RSI mean reversion or trend change.
# Designed for low trade frequency (<50/year) with clear entry/exit rules to avoid overtrading.

name = "4h_RSI_Trend_Momentum_Volume"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 70 (bullish momentum) + price above 1d EMA50 (uptrend) + volume spike
            if (rsi[i] > 70 and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 30 (bearish momentum) + price below 1d EMA50 (downtrend) + volume spike
            elif (rsi[i] < 30 and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 50 (loss of momentum) or price below 1d EMA50 (trend change)
            if (rsi[i] < 50 or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 50 (loss of momentum) or price above 1d EMA50 (trend change)
            if (rsi[i] > 50 or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 01:40
