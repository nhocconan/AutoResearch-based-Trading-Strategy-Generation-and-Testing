# Strategy: 4h_Donchian20_12hEMA50_VolumeConfirm_Balanced

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.031 | +20.1% | -17.8% | 147 | PASS |
| ETHUSDT | 0.175 | +29.8% | -16.7% | 139 | PASS |
| SOLUSDT | 1.017 | +201.3% | -22.4% | 144 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.960 | -5.0% | -8.6% | 52 | FAIL |
| ETHUSDT | 0.146 | +7.7% | -8.9% | 46 | PASS |
| SOLUSDT | 0.174 | +8.2% | -12.7% | 48 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation
# Uses Donchian channel breakouts from 4h for clear entry/exit structure
# 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (>1.3x 20 EMA volume) filters false breakouts
# Discrete sizing 0.25 minimizes fee churn while targeting 75-200 total trades over 4 years
# Works in bull markets (continuation at upper band) and bear markets (continuation at lower band)
# Focus on BTC/ETH by requiring 12h trend alignment (avoids SOL-only bias)

name = "4h_Donchian20_12hEMA50_VolumeConfirm_Balanced"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) trend filter from prior completed 12h bar
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_shifted = np.roll(ema_50_12h, 1)
    ema_50_12h_shifted[0] = np.nan
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_shifted)
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels (20) from prior completed 4h bar
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Shift to use prior completed 4h bar
    upper_20_shifted = np.roll(upper_20, 1)
    lower_20_shifted = np.roll(lower_20, 1)
    upper_20_shifted[0] = np.nan
    lower_20_shifted[0] = np.nan
    
    # Align to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20_shifted)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND price > 12h EMA50 AND volume spike
            if close[i] > upper_20_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > (1.3 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian AND price < 12h EMA50 AND volume spike
            elif close[i] < lower_20_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > (1.3 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower Donchian OR price crosses below 12h EMA50
            if close[i] < lower_20_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper Donchian OR price crosses above 12h EMA50
            if close[i] > upper_20_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 10:50
