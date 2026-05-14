# Strategy: 6h_Ichimoku_Cloud_DailyTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.124 | +15.2% | -17.6% | 103 | DISCARD |
| ETHUSDT | 0.282 | +35.1% | -9.2% | 99 | KEEP |
| SOLUSDT | 1.073 | +169.2% | -16.8% | 103 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.421 | +11.7% | -8.7% | 24 | KEEP |
| SOLUSDT | -1.190 | -13.6% | -17.1% | 33 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
6h Ichimoku Cloud with Daily Trend Filter
Long: Tenkan > Kijun + price above Kumo (cloud) + Kijun rising (1d)
Short: Tenkan < Kijun + price below Kumo + Kijun falling (1d)
Exit: Opposite TK cross or price crosses Kijun
Uses Ichimoku on 6h for entry timing, 1d Kijun slope for trend filter.
Designed to capture trend continuations in both bull and bear markets.
Target: 60-120 total trades over 4 years (15-30/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A, Senkou B"""
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 52 periods
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku on 6h
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Get 1d data for trend filter (Kijun slope)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Kijun
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Align 1d Kijun to 6h
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    
    # Calculate Kijun slope (1-period change) for trend filter
    kijun_slope = np.diff(kijun_1d_aligned, prepend=kijun_1d_aligned[0])
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 52  # need Ichimoku calculations
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(kijun_1d_aligned[i]) or np.isnan(kijun_slope[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        price = close[i]
        
        if position == 0:
            # Long: TK bullish + price above cloud + rising 1d Kijun
            if tenkan[i] > kijun[i] and price > cloud_top and kijun_slope[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: TK bearish + price below cloud + falling 1d Kijun
            elif tenkan[i] < kijun[i] and price < cloud_bottom and kijun_slope[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK bearish OR price crosses below Kijun
            if tenkan[i] < kijun[i] or price < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK bullish OR price crosses above Kijun
            if tenkan[i] > kijun[i] or price > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_DailyTrend"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-18 00:00
