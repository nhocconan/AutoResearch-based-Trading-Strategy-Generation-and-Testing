# Strategy: 4h_TripleBarrier_Breakout_12hTrend_VolumeConfirmation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.310 | +6.5% | -20.0% | 188 | FAIL |
| ETHUSDT | 0.498 | +51.4% | -14.2% | 171 | PASS |
| SOLUSDT | 0.430 | +58.0% | -26.7% | 168 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.253 | +9.3% | -8.0% | 64 | PASS |
| SOLUSDT | 0.459 | +13.0% | -9.1% | 55 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_TripleBarrier_Breakout_12hTrend_VolumeConfirmation
# Hypothesis: Combine Donchian breakout with Bollinger squeeze release and 12h trend filter.
# This creates high-conviction entries during volatility expansion in trending markets,
# reducing false breakouts. Volatility contraction (squeeze) followed by expansion
# captures meaningful moves while avoiding chop. Works in bull/bear by following 12h trend.
# Target: 20-30 trades/year (80-120 total) to minimize fee drag.

name = "4h_TripleBarrier_Breakout_12hTrend_VolumeConfirmation"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)

    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values

    # Calculate 12h EMA30 for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    # Calculate Donchian channels (20-period) on 4h
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])

    # Bollinger Bands (20,2) for squeeze detection
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma + 2 * std
    lower_bb = ma - 2 * std
    bb_width = (upper_bb - lower_bb) / ma  # normalized width

    # Bollinger squeeze: width below 20-period 10th percentile
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).quantile(0.1).values
    squeeze = bb_width < bb_width_percentile

    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(squeeze[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian + BB squeeze release + 12h uptrend + volume spike
            if (close[i] > highest_high[i] and 
                not squeeze[i] and  # squeeze released (volatility expanding)
                close[i] > ema_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + BB squeeze release + 12h downtrend + volume spike
            elif (close[i] < lowest_low[i] and 
                  not squeeze[i] and 
                  close[i] < ema_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Donchian channel or 12h trend turns down
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] < midpoint or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Donchian channel or 12h trend turns up
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] > midpoint or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 05:41
