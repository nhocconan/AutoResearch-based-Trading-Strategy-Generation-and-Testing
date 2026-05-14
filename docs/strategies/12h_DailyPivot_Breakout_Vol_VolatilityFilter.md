# Strategy: 12h_DailyPivot_Breakout_Vol_VolatilityFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.024 | +21.4% | -8.2% | 68 | PASS |
| ETHUSDT | 0.035 | +21.2% | -12.5% | 59 | PASS |
| SOLUSDT | 0.508 | +65.7% | -26.3% | 55 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.869 | -2.1% | -9.6% | 26 | FAIL |
| ETHUSDT | 0.551 | +14.3% | -6.8% | 24 | PASS |
| SOLUSDT | -1.324 | -13.8% | -22.1% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 12h timeframe (use previous day's levels)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate weekly ATR for volatility filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr_1w = np.maximum(high_1w - low_1w, np.maximum(abs(high_1w - np.roll(close_1w, 1)), abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr14_12h = align_htf_to_ltf(prices, df_1w, atr14_1w, additional_delay_bars=1)
    
    # Volume filter: current volume > 2.0 * 24-period average (24 periods = 12 days at 12h)
    volume_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 24  # Need sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(atr14_12h[i]) or np.isnan(volume_ma24[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (2.0 * volume_ma24[i])
        
        # ATR filter: only trade when volatility is above average
        vol_filter = atr14_12h[i] > 0
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and volatility
            if (close[i] > r1_12h[i] and volume_filter and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and volatility
            elif (close[i] < s1_12h[i] and volume_filter and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S1 or volatility drops
            if close[i] < s1_12h[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R1 or volatility drops
            if close[i] > r1_12h[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyPivot_Breakout_Vol_VolatilityFilter"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-17 12:10
