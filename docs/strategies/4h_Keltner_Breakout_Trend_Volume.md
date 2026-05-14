# Strategy: 4h_Keltner_Breakout_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.148 | +13.4% | -12.2% | 111 | FAIL |
| ETHUSDT | 0.301 | +36.8% | -10.7% | 103 | PASS |
| SOLUSDT | 0.490 | +60.7% | -23.7% | 98 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.411 | +11.7% | -6.3% | 37 | PASS |
| SOLUSDT | 0.200 | +8.5% | -13.7% | 33 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Keltner_Breakout_Trend_Volume
# Hypothesis: Price breaking out of Keltner Channels with trend confirmation and volume spike captures strong momentum moves in both bull and bear markets.
# Uses exponential moving average with ATR-based bands for adaptive volatility.
# Entry: Long when close > upper band + EMA50 uptrend + volume spike; Short when close < lower band + EMA50 downtrend + volume spike.
# Exit: Mean reversion to EMA50 (middle band) to avoid overstaying in extended moves.
# Target: 25-35 trades/year on 4h to stay within optimal range while capturing significant moves.

name = "4h_Keltner_Breakout_Trend_Volume"
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

    # Calculate ATR for Keltner Channels
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # EMA20 as middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Keltner Channels: upper = EMA20 + 2*ATR, lower = EMA20 - 2*ATR
    upper_band = ema20 + 2.0 * atr
    lower_band = ema20 - 2.0 * atr

    # EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume confirmation: volume > 1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema50[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above upper band + EMA50 uptrend + volume spike
            if (close[i] > upper_band[i] and 
                close[i] > ema50[i] and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below lower band + EMA50 downtrend + volume spike
            elif (close[i] < lower_band[i] and 
                  close[i] < ema50[i] and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to EMA20 (middle band)
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion to EMA20 (middle band)
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 00:29
