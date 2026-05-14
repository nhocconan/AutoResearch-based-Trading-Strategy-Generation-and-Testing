# Strategy: 4h_DailyPivot_Breakout_Vol_VolatilityFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.355 | +40.8% | -13.5% | 229 | PASS |
| ETHUSDT | 0.100 | +24.0% | -17.6% | 220 | PASS |
| SOLUSDT | 0.598 | +89.5% | -24.1% | 208 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.791 | -3.8% | -13.8% | 82 | FAIL |
| ETHUSDT | 0.673 | +18.8% | -13.3% | 72 | PASS |
| SOLUSDT | 0.346 | +12.2% | -19.4% | 79 | PASS |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 4h timeframe (use previous day's levels)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get weekly data for volatility filter (ATR)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr_1w = np.maximum(high_1w - low_1w, np.maximum(abs(high_1w - np.roll(close_1w, 1)), abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr14_4h = align_htf_to_ltf(prices, df_1w, atr14_1w, additional_delay_bars=1)
    
    # Volume filter: current volume > 1.5 * 20-period average (20 periods = 10 days at 4h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(atr14_4h[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr14_4h[i] > 0
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and volatility
            if (close[i] > r1_4h[i] and volume_filter and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and volatility
            elif (close[i] < s1_4h[i] and volume_filter and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S1 or volatility drops
            if close[i] < s1_4h[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R1 or volatility drops
            if close[i] > r1_4h[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyPivot_Breakout_Vol_VolatilityFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 12:10
