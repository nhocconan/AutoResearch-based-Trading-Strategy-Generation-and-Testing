# Strategy: 6h_Donchian20_12hEMA50_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.329 | +1.1% | -17.9% | 93 | FAIL |
| ETHUSDT | 0.086 | +22.7% | -13.3% | 88 | PASS |
| SOLUSDT | 1.158 | +235.1% | -21.3% | 74 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.374 | +12.4% | -9.0% | 28 | PASS |
| SOLUSDT | -0.190 | +0.3% | -18.0% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>1.6x 20 EMA volume)
# Uses 6h Donchian channel breakouts for structure - captures strong momentum bursts
# 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>1.6x average volume) - tighter to reduce trades
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 75-200 total trades over 4 years = 19-50/year for 6h timeframe
# Works in bull markets (continuation at upper channel) and bear markets (continuation at lower channel)
# Focus on BTC/ETH by requiring 12h trend alignment (avoids SOL-only bias)

name = "6h_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) trend filter from prior completed 12h bar
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_shifted = np.roll(ema_50_12h, 1)
    ema_50_12h_shifted[0] = np.nan
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 6h Donchian channels (20-period) from prior completed 6h bar
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Upper channel: 20-period high
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only prior completed 6h bar (no look-ahead)
    upper_channel_shifted = np.roll(upper_channel, 1)
    lower_channel_shifted = np.roll(lower_channel, 1)
    upper_channel_shifted[0] = np.nan
    lower_channel_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(upper_channel_shifted[i]) or np.isnan(lower_channel_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper channel AND price > 12h EMA50 AND volume spike
            if close[i] > upper_channel_shifted[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > (1.6 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower channel AND price < 12h EMA50 AND volume spike
            elif close[i] < lower_channel_shifted[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > (1.6 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower channel OR price crosses below 12h EMA50
            if close[i] < lower_channel_shifted[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper channel OR price crosses above 12h EMA50
            if close[i] > upper_channel_shifted[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 10:22
