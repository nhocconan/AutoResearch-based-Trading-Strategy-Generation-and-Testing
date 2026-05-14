# Strategy: 4h_12h_donchian_ema50_volume_filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.225 | +32.8% | -14.9% | 85 | KEEP |
| ETHUSDT | 0.001 | +15.7% | -15.8% | 83 | KEEP |
| SOLUSDT | 0.978 | +195.2% | -22.9% | 84 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.148 | -7.5% | -13.5% | 31 | DISCARD |
| ETHUSDT | 0.242 | +9.7% | -9.8% | 29 | KEEP |
| SOLUSDT | -0.278 | -1.7% | -16.7% | 28 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h Donchian breakout + volume confirmation + 12h EMA trend filter
# Long when price breaks above 12h Donchian upper channel with volume > 1.3x average and price > 12h EMA50
# Short when price breaks below 12h Donchian lower channel with volume > 1.3x average and price < 12h EMA50
# Exit on opposite Donchian breakout or when price crosses 12h EMA50
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and cost
# Uses 12h timeframe for structure to reduce noise and false breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 12h indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume surge condition
        volume_surge = volume[i] > 1.3 * vol_ma_20[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # EMA trend filter
        above_ema = close[i] > ema50_aligned[i]
        below_ema = close[i] < ema50_aligned[i]
        
        # Entry logic: breakout + volume + trend filter
        long_entry = long_breakout and volume_surge and above_ema
        short_entry = short_breakout and volume_surge and below_ema
        
        # Exit conditions: opposite breakout or EMA cross
        exit_long = position == 1 and (short_breakout or close[i] < ema50_aligned[i])
        exit_short = position == -1 and (long_breakout or close[i] > ema50_aligned[i])
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_ema50_volume_filter_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 13:04
