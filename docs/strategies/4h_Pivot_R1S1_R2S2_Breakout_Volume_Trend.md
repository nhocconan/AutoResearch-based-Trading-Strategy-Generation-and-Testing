# Strategy: 4h_Pivot_R1S1_R2S2_Breakout_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.136 | +26.2% | -11.9% | 372 | PASS |
| ETHUSDT | 0.130 | +26.2% | -11.6% | 341 | PASS |
| SOLUSDT | 0.603 | +72.6% | -24.3% | 277 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.823 | -8.6% | -13.4% | 147 | FAIL |
| ETHUSDT | 0.643 | +14.9% | -6.1% | 125 | PASS |
| SOLUSDT | 0.218 | +8.6% | -11.8% | 100 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Pivot_R1S1_R2S2_Breakout_Volume_Trend
Hypothesis: Price breaks above/below S2/R2 levels with volume confirmation and 4h EMA20 trend filter.
Uses 1d Camarilla pivot levels (R1,S1,R2,S2), volume > 2x 20-period average, and EMA20 trend filter.
Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag while capturing institutional breakout moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels (calculated from previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1, R2, S2 based on previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # R2 = close + 1.1*(high-low)/6
    # S2 = close - 1.1*(high-low)/6
    camarilla_range = (high_1d - low_1d) * 1.1 / 6
    r1 = close_1d + camarilla_range / 2
    s1 = close_1d - camarilla_range / 2
    r2 = close_1d + camarilla_range
    s2 = close_1d - camarilla_range
    
    # Align to 4h timeframe (use previous day's levels for current day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # EMA20 for trend filter on 4h
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike: >2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 1)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_20[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        ema20 = ema_20[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above S2 with volume spike in uptrend
            if (price > s2_level and          # breaks above S2
                vol_spike and                 # volume confirmation
                price > ema20):               # uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R2 with volume spike in downtrend
            elif (price < r2_level and        # breaks below R2
                  vol_spike and               # volume confirmation
                  price < ema20):             # downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses back below S1 or trend reverses
            if price < s1_level or price < ema20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses back above R1 or trend reverses
            if price > r1_level or price > ema20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Pivot_R1S1_R2S2_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 03:45
