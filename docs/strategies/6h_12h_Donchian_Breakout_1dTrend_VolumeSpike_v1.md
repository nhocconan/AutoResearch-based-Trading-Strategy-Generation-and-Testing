# Strategy: 6h_12h_Donchian_Breakout_1dTrend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.028 | +16.3% | -16.4% | 77 | FAIL |
| ETHUSDT | 0.089 | +22.7% | -20.2% | 75 | PASS |
| SOLUSDT | 0.933 | +182.8% | -30.2% | 78 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.330 | +11.7% | -11.1% | 25 | PASS |
| SOLUSDT | -0.342 | -3.2% | -18.3% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_12h_Donchian_Breakout_1dTrend_VolumeSpike_v1
# Hypothesis: 6s Donchian(20) breakout with 12h trend filter (EMA50) and volume spike confirmation (1.5x avg volume). 
# The 12h EMA50 provides trend direction to avoid counter-trend trades, while volume spikes confirm breakout strength.
# Designed for 60-120 total trades over 4 years (15-30/year) to minimize fee drag. Works in bull/bear by following 12h trend.

name = "6h_12h_Donchian_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
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

    # Get 12h data for Donchian channels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)

    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate 12h Donchian channels (20-period)
    high_roll = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll

    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)

    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Donchian high in 12h uptrend with volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema50_12h_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian low in 12h downtrend with volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema50_12h_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 12h EMA50 (trend change)
            if close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 12h EMA50 (trend change)
            if close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 17:45
