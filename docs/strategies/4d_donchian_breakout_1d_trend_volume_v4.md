# Strategy: 4d_donchian_breakout_1d_trend_volume_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.172 | +5.1% | -29.7% | 141 | FAIL |
| ETHUSDT | 0.007 | +14.2% | -18.9% | 131 | PASS |
| SOLUSDT | 0.878 | +196.8% | -28.8% | 137 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.091 | +6.3% | -11.4% | 47 | PASS |
| SOLUSDT | 0.496 | +16.9% | -11.9% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4d_donchian_breakout_1d_trend_volume_v4
Hypothesis: Breakouts from Donchian(20) channel on 4h filtered by daily EMA20 trend and volume spike (>1.5x average).
Long when price breaks above upper Donchian with volume spike and price above daily EMA20.
Short when price breaks below lower Donchian with volume spike and price below daily EMA20.
Designed for ~25-35 trades/year on 4h with strict entry conditions to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4d_donchian_breakout_1d_trend_volume_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average (spike)
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Donchian breakout conditions
        bullish_breakout = close[i] > high_max[i-1]
        bearish_breakout = close[i] < low_min[i-1]
        
        # Daily trend filter
        above_1d_ema20 = close[i] > ema20_1d_aligned[i]
        below_1d_ema20 = close[i] < ema20_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: bearish breakout or trend turns bearish
            if bearish_breakout or below_1d_ema20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: bullish breakout or trend turns bullish
            if bullish_breakout or above_1d_ema20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long: bullish Donchian breakout with volume spike and bullish trend
            if bullish_breakout and vol_spike and above_1d_ema20:
                position = 1
                signals[i] = 0.30
            # Short: bearish Donchian breakout with volume spike and bearish trend
            elif bearish_breakout and vol_spike and below_1d_ema20:
                position = -1
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-04-07 20:50
