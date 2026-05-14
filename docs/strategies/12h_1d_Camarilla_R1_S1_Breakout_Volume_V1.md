# Strategy: 12h_1d_Camarilla_R1_S1_Breakout_Volume_V1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.055 | +22.4% | -16.3% | 199 | PASS |
| ETHUSDT | 0.024 | +20.0% | -15.9% | 174 | PASS |
| SOLUSDT | 0.146 | +26.8% | -28.3% | 158 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.573 | -9.1% | -13.8% | 72 | FAIL |
| ETHUSDT | 0.355 | +11.1% | -9.1% | 60 | PASS |
| SOLUSDT | -0.726 | -6.4% | -18.4% | 53 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_R1_S1_Breakout_Volume_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (using previous day's data)
    def calculate_camarilla(high_arr, low_arr, close_arr):
        n_days = len(close_arr)
        R1 = np.full(n_days, np.nan)
        S1 = np.full(n_days, np.nan)
        
        for i in range(1, n_days):
            # Use previous day's OHLC
            high_prev = high_arr[i-1]
            low_prev = low_arr[i-1]
            close_prev = close_arr[i-1]
            
            # Camarilla formulas
            R1[i] = close_prev + (high_prev - low_prev) * 1.1 / 12
            S1[i] = close_prev - (high_prev - low_prev) * 1.1 / 12
        
        return R1, S1
    
    R1, S1 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate volume spike indicator (volume > 1.5 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation required
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when price breaks above R1 with volume
            if close[i] > R1_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume
            elif close[i] < S1_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below S1 (reversal)
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above R1 (reversal)
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 16:10
