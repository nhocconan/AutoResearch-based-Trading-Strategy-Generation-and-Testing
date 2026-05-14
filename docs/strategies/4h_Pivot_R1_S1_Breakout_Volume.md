# Strategy: 4h_Pivot_R1_S1_Breakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.261 | +3.9% | -23.7% | 390 | FAIL |
| ETHUSDT | 0.094 | +23.4% | -21.2% | 372 | PASS |
| SOLUSDT | 0.667 | +106.5% | -27.7% | 328 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.044 | +5.7% | -15.5% | 123 | PASS |
| SOLUSDT | -0.111 | +2.4% | -16.0% | 118 | FAIL |

## Code
```python
# 4h_Pivot_R1_S1_Breakout_Volume
# Hypothesis: Breakouts above/below daily R1/S1 levels with volume confirmation capture breakouts in trending markets while avoiding false moves during consolidation. The volume filter reduces false breakouts, and the session filter focuses on active trading hours. This strategy works in both bull and bear markets because it captures directional moves regardless of overall trend, relying on price action at key pivot levels.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Pivot_R1_S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivot levels (R1/S1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (R1, S1) from previous daily bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    R1 = pivot + (range_hl * 1.1 / 12)
    S1 = pivot - (range_hl * 1.1 / 12)
    
    # Align R1/S1 to 4h (wait for daily close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        vol_filter = volume_filter[i]
        sess_filter = session_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume and session
            if close_val > R1_val and vol_filter and sess_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and session
            elif close_val < S1_val and vol_filter and sess_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below S1
            if close_val < S1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above R1
            if close_val > R1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-18 23:11
