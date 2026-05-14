# Strategy: 12H_Camarilla_R1S1_VolumeConfirmation_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.558 | -1.4% | -21.1% | 239 | FAIL |
| ETHUSDT | 0.073 | +23.2% | -21.5% | 203 | PASS |
| SOLUSDT | -0.207 | -0.2% | -39.7% | 190 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.101 | +6.9% | -9.2% | 76 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d Camarilla R1/S1 breakout with volume confirmation and ATR stoploss.
Long when price breaks above 1d Camarilla R1 AND volume > 1.5x 20-period average.
Short when price breaks below 1d Camarilla S1 AND volume > 1.5x 20-period average.
Exit when price retouches 1d Camarilla pivot point or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to balance return and drawdown.
Designed for 12h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Works in both bull and bear markets by using volume confirmation to filter false breakouts and ATR stops to manage risk.
1d Camarilla levels provide institutional support/resistance from higher timeframe.
"""

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
    
    # Calculate 1d Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day's OHLC)
    camarilla_h1 = (high_1d + low_1d + close_1d) / 3.0
    camarilla_l1 = (high_1d + low_1d + close_1d) / 3.0
    camarilla_range = high_1d - low_1d
    
    camarilla_r1 = camarilla_h1 + camarilla_range * 1.1 / 12.0
    camarilla_s1 = camarilla_l1 - camarilla_range * 1.1 / 12.0
    camarilla_pivot = camarilla_h1  # Pivot point
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1d Camarilla R1 AND volume spike
            if (price > r1 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below 1d Camarilla S1 AND volume spike
            elif (price < s1 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches 1d Camarilla pivot point
            if position == 1 and price <= pivot:
                exit_signal = True
            elif position == -1 and price >= pivot:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_VolumeConfirmation_ATRStop"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 05:48
