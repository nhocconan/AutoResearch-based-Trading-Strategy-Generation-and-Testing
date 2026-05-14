# Strategy: 4h_Camarilla_L4H4_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.301 | +35.9% | -11.2% | 145 | KEEP |
| ETHUSDT | 0.072 | +22.4% | -14.9% | 143 | KEEP |
| SOLUSDT | 0.996 | +168.7% | -22.6% | 128 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.956 | -4.7% | -9.0% | 54 | DISCARD |
| ETHUSDT | 0.796 | +20.4% | -9.7% | 48 | KEEP |
| SOLUSDT | 0.286 | +10.3% | -13.7% | 41 | KEEP |

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
    
    # Get 1d data for ATR and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR using Wilder's smoothing
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_1d[i] = np.mean(tr[1:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Get 1d data for Camarilla pivot calculation
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift[0] = np.nan
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    
    # Calculate Camarilla levels for previous day
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d_shift[i]) or np.isnan(low_1d_shift[i]) or np.isnan(close_1d_shift[i])):
            camarilla_h4[i] = close_1d_shift[i] + 1.1 * (high_1d_shift[i] - low_1d_shift[i]) / 2
            camarilla_l4[i] = close_1d_shift[i] - 1.1 * (high_1d_shift[i] - low_1d_shift[i]) / 2
    
    # Get 1d EMA34 for trend filter
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                        ema_1d[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR, Camarilla, EMA, and volume MA
    start_idx = max(14, 34, vol_period) + 20  # extra buffer for ATR calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr = atr_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above H4 with volume + price > EMA34
            if (price > camarilla_h4_aligned[i] and 
                vol_ratio > 2.0 and 
                price > ema_1d_aligned[i]):
                signals[i] = size
                position = 1
            # Short: Price breaks below L4 with volume + price < EMA34
            elif (price < camarilla_l4_aligned[i] and 
                  vol_ratio > 2.0 and 
                  price < ema_1d_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below L4 or ATR-based stop
            if (price < camarilla_l4_aligned[i] or 
                price < close[i-1] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above H4 or ATR-based stop
            if (price > camarilla_h4_aligned[i] or 
                price > close[i-1] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_L4H4_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 12:06
