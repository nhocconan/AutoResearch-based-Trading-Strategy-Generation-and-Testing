# Strategy: 6h_Donchian20_Breakout_1dEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.233 | +8.1% | -18.1% | 67 | FAIL |
| ETHUSDT | 0.040 | +20.1% | -13.9% | 69 | PASS |
| SOLUSDT | 0.803 | +125.6% | -19.1% | 60 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.359 | +11.9% | -8.5% | 25 | PASS |
| SOLUSDT | -0.483 | -4.8% | -19.4% | 21 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter and volume confirmation
# Uses 1d EMA50 for HTF trend alignment to reduce whipsaw vs shorter timeframes
# Donchian levels from 6h provide clear structure for breakouts
# Breakout at upper/lower Donchian with volume spike confirms institutional participation
# 1d EMA50 trend filter ensures alignment with daily trend
# Works in both bull and bear markets by following 1d trend
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Discrete position sizing: 0.25 (25% of capital) to minimize fee churn

name = "6h_Donchian20_Breakout_1dEMA50_Volume"
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
    
    # Calculate 6h Donchian levels (20-period)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian channels: upper = max(high, 20), lower = min(low, 20)
    upper_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_6h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_6h, lower_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.0x 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper with volume spike AND price > 1d EMA50 (bullish trend)
            if (close[i] > upper_20_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower with volume spike AND price < 1d EMA50 (bearish trend)
            elif (close[i] < lower_20_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian lower OR below 1d EMA50 (trend change)
            if close[i] < lower_20_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper OR above 1d EMA50 (trend change)
            if close[i] > upper_20_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 09:53
