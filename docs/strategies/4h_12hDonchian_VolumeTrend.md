# Strategy: 4h_12hDonchian_VolumeTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.166 | +9.0% | -17.5% | 48 | FAIL |
| ETHUSDT | 0.126 | +25.7% | -16.4% | 46 | PASS |
| SOLUSDT | 0.519 | +79.3% | -27.4% | 44 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.147 | +7.7% | -17.7% | 18 | PASS |
| SOLUSDT | -0.016 | +3.8% | -17.7% | 15 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and 12h EMA trend filter.
# Designed for low trade frequency (15-25/year) to avoid fee drag. Uses 12h structure for trend direction,
# 4h volume surge for momentum confirmation, and 12h EMA for trend alignment.
# Works in bull/bear markets by following higher timeframe trends with strict entry filters.

name = "4h_12hDonchian_VolumeTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channel and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channel (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donchian_high = rolling_max(high_12h, 20)
    donchian_low = rolling_min(low_12h, 20)
    
    # Calculate 12h EMA (50-period)
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 4h data for volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Volume spike: 2x 20-period EMA
    vol_ema = pd.Series(volume_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume_4h > (vol_ema * 2.0)
    
    # Align volume spike to 4h timeframe (already 4h, but ensure alignment)
    vol_spike_aligned = align_htf_to_ltf(prices, df_4h, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 12h Donchian high + volume surge + above 12h EMA
            if close[i] > donchian_high_aligned[i] and vol_spike_aligned[i] and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 12h Donchian low + volume surge + below 12h EMA
            elif close[i] < donchian_low_aligned[i] and vol_spike_aligned[i] and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 12h Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 12h Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 17:03
