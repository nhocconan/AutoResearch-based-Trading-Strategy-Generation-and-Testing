# Strategy: 12h_camarilla_pivot_1d_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.006 | +18.7% | -13.0% | 75 | FAIL |
| ETHUSDT | -0.279 | -0.7% | -16.9% | 76 | FAIL |
| SOLUSDT | 0.578 | +88.6% | -32.5% | 62 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.135 | +7.4% | -13.6% | 23 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
12h Camarilla Pivot with 1-day Trend Filter and Volume Confirmation
Hypothesis: Camarilla pivot levels from daily charts act as strong support/resistance in 12h timeframe.
Combined with 1-day EMA trend filter and volume confirmation to avoid false breakouts.
Designed for ~20-30 trades/year to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day
    # H = high, L = low, C = close of previous day
    H = high_1d
    L = low_1d
    C = close_1d
    
    # Camarilla levels
    R4 = C + ((H - L) * 1.500)
    R3 = C + ((H - L) * 1.250)
    R2 = C + ((H - L) * 1.166)
    R1 = C + ((H - L) * 1.083)
    S1 = C - ((H - L) * 1.083)
    S2 = C - ((H - L) * 1.166)
    S3 = C - ((H - L) * 1.250)
    S4 = C - ((H - L) * 1.500)
    
    # Align Camarilla levels to 12h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(R2_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(S2_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(S4_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S1 OR trend reverses
            if (close[i] < S1_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R1 OR trend reverses
            if (close[i] > R1_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 1-day EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Long: price breaks above R1 with uptrend and volume spike
            if (high[i] > R1_aligned[i-1] and 
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S1 with downtrend and volume spike
            elif (low[i] < S1_aligned[i-1] and 
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 01:52
