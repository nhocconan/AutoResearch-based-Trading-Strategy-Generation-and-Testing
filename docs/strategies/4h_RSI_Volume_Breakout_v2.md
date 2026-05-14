# Strategy: 4h_RSI_Volume_Breakout_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.056 | +16.9% | -11.0% | 106 | FAIL |
| ETHUSDT | 0.361 | +41.8% | -11.2% | 98 | PASS |
| SOLUSDT | 0.538 | +70.2% | -19.7% | 89 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.326 | +10.3% | -8.0% | 32 | PASS |
| SOLUSDT | -0.376 | -0.9% | -18.7% | 32 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_RSI_Volume_Breakout_v2
# Hypothesis: RSI(14) extremes with volume confirmation and 12h EMA trend filter on 4h timeframe.
# Uses 12h EMA50 as trend filter to avoid counter-trend trades, reducing false signals and trade frequency.
# RSI > 70 with volume spike and price above 12h EMA50 = long
# RSI < 30 with volume spike and price below 12h EMA50 = short
# Exits when RSI crosses 50 (momentum fade)
# Designed for 20-40 trades/year to minimize fee drag. Works in both bull and bear by capturing momentum bursts with trend alignment.

name = "4h_RSI_Volume_Breakout_v2"
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

    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma)

    # Get 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if RSI or volume data is not ready
        if i < 14 or np.isnan(rsi[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 70 (overbought momentum) with volume spike and price above 12h EMA50 (uptrend)
            if rsi[i] > 70 and volume_spike[i] and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 30 (oversold capitulation) with volume spike and price below 12h EMA50 (downtrend)
            elif rsi[i] < 30 and volume_spike[i] and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI falls below 50 (momentum fading)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI rises above 50 (selling pressure fading)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 04:55
