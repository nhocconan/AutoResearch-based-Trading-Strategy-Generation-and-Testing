# Strategy: 6h_BollingerBreakout_Pullback_1dTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.234 | +31.4% | -14.4% | 133 | PASS |
| ETHUSDT | 0.525 | +53.4% | -9.5% | 131 | PASS |
| SOLUSDT | 0.732 | +102.6% | -24.5% | 117 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.390 | -7.6% | -9.7% | 51 | FAIL |
| ETHUSDT | 0.896 | +21.2% | -6.4% | 41 | PASS |
| SOLUSDT | 0.277 | +10.0% | -10.1% | 38 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_BollingerBreakout_Pullback_1dTrend
# Hypothesis: Bollinger Band breakout with pullback entry on 6h, filtered by 1d trend and volume confirmation.
# This strategy aims to capture trend continuation after pullbacks in trending markets while avoiding false breakouts in ranging conditions.
# Works in both bull and bear markets by only trading in the direction of the 1d trend.

name = "6h_BollingerBreakout_Pullback_1dTrend"
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

    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')

    # Bollinger Bands (20, 2) on 6h
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)

    # Bollinger Band breakout signals (when price crosses outside bands)
    bb_breakout_up = close > upper_bb
    bb_breakout_down = close < lower_bb

    # Pullback condition: price returns inside bands after breakout
    # We track if a breakout occurred and price has since returned inside
    breakout_up_active = np.zeros(n, dtype=bool)
    breakout_down_active = np.zeros(n, dtype=bool)

    for i in range(1, n):
        # Carry forward breakout state
        breakout_up_active[i] = breakout_up_active[i-1]
        breakout_down_active[i] = breakout_down_active[i-1]

        # Activate breakout state when price breaks outside bands
        if bb_breakout_up[i]:
            breakout_up_active[i] = True
        if bb_breakout_down[i]:
            breakout_down_active[i] = True

        # Deactivate breakout state when price returns inside bands
        if breakout_up_active[i] and (lower_bb[i] <= close[i] <= upper_bb[i]):
            breakout_up_active[i] = False
        if breakout_down_active[i] and (lower_bb[i] <= close[i] <= upper_bb[i]):
            breakout_down_active[i] = False

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish breakout pullback in uptrend with volume
            if (breakout_up_active[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish breakout pullback in downtrend with volume
            elif (breakout_down_active[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Bollinger middle or trend turns down
            if close[i] < sma20[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Bollinger middle or trend turns up
            if close[i] > sma20[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 04:27
