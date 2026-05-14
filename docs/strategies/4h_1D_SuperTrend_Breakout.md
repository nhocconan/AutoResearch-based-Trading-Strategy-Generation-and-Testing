# Strategy: 4h_1D_SuperTrend_Breakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.186 | +29.9% | -13.9% | 208 | PASS |
| ETHUSDT | 0.214 | +33.0% | -13.5% | 204 | PASS |
| SOLUSDT | 0.866 | +157.0% | -32.1% | 195 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.670 | -2.3% | -7.6% | 77 | FAIL |
| ETHUSDT | 0.298 | +10.8% | -12.5% | 76 | PASS |
| SOLUSDT | 0.232 | +9.6% | -11.8% | 75 | PASS |

## Code
```python
#!/usr/bin/env python3

# 4h_1D_SuperTrend_Breakout
# Hypothesis: Breakout above/below 4-period ATR SuperTrend on 4h with 1d EMA trend filter and volume confirmation.
# SuperTrend adapts to volatility, providing dynamic support/resistance. Works in both bull and bear markets
# by requiring trend alignment and volume confirmation to avoid false breakouts. Targets 20-40 trades/year.

name = "4h_1D_SuperTrend_Breakout"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate 4h SuperTrend (10-period ATR, multiplier 3.0)
    atr_period = 10
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    hl2 = (high + low) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)

    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend

    supertrend[0] = upper_band[0]
    direction[0] = 1

    for i in range(1, n):
        if close[i-1] > supertrend[i-1]:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])

        if close[i] > supertrend[i]:
            direction[i] = 1
        else:
            direction[i] = -1

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(supertrend[i]) or
            np.isnan(direction[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1d
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Price above SuperTrend with bullish trend and volume confirmation
            if close[i] > supertrend[i] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below SuperTrend with bearish trend and volume confirmation
            elif close[i] < supertrend[i] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below SuperTrend or trend turns bearish
            if close[i] < supertrend[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above SuperTrend or trend turns bullish
            if close[i] > supertrend[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 16:45
