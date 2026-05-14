# Strategy: 4H_Camarilla_H3L3_VolumeConfirmation_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.000 | +20.5% | -10.2% | 249 | FAIL |
| ETHUSDT | 0.418 | +42.5% | -9.1% | 235 | PASS |
| SOLUSDT | 0.573 | +69.8% | -18.5% | 203 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.351 | +10.2% | -8.7% | 84 | PASS |
| SOLUSDT | 0.675 | +15.2% | -9.1% | 68 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Camarilla H3/L3 breakout with volume confirmation and ATR stoploss.
Long when price breaks above 1d Camarilla H3 AND volume > 1.8x 20-period average.
Short when price breaks below 1d Camarilla L3 AND volume > 1.8x 20-period average.
Exit when price retouches 1d Camarilla pivot point or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to balance return and drawdown.
Designed for 4h timeframe to target 19-50 trades/year per symbol (75-200 total over 4 years).
Works in both bull and bear markets by using volume confirmation to filter false breakouts and ATR stops to manage risk.
1d Camarilla H3/L3 levels provide stronger institutional support/resistance from higher timeframe.
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
    camarilla_h3 = high_1d + (high_1d - low_1d) * 1.1 / 4.0
    camarilla_l3 = low_1d - (high_1d - low_1d) * 1.1 / 4.0
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume average (20-period) on 4h timeframe
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
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1d Camarilla H3 AND volume spike
            if (price > h3 and volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below 1d Camarilla L3 AND volume spike
            elif (price < l3 and volume[i] > 1.8 * vol_ma_val):
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

name = "4H_Camarilla_H3L3_VolumeConfirmation_ATRStop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 05:51
