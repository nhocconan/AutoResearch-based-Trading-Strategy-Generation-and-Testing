# Strategy: 6h_1d_Pivot_S1R1_EMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.477 | +41.1% | -5.6% | 204 | PASS |
| ETHUSDT | 0.273 | +33.8% | -10.9% | 168 | PASS |
| SOLUSDT | 0.843 | +108.0% | -16.8% | 146 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.968 | -2.8% | -6.0% | 77 | FAIL |
| ETHUSDT | 0.895 | +19.9% | -8.3% | 64 | PASS |
| SOLUSDT | -0.018 | +5.1% | -7.2% | 58 | FAIL |

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
    
    # Load daily data for pivots and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA50 for trend
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Calculate daily pivot points (using prior day's OHLC)
    pivot_point = np.full_like(close_1d, np.nan)
    resistance1 = np.full_like(close_1d, np.nan)
    support1 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 2:
        for i in range(1, len(close_1d)):
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            
            pp = (ph + pl + pc) / 3.0
            r1 = 2 * pp - pl
            s1 = 2 * pp - ph
            
            pivot_point[i] = pp
            resistance1[i] = r1
            support1[i] = s1
    
    # Align 1d indicators to 6h timeframe
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    pivot_point_6h = align_htf_to_ltf(prices, df_1d, pivot_point)
    resistance1_6h = align_htf_to_ltf(prices, df_1d, resistance1)
    support1_6h = align_htf_to_ltf(prices, df_1d, support1)
    
    # Volume spike detection on 6h bars
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_6h[i]) or 
            np.isnan(pivot_point_6h[i]) or 
            np.isnan(resistance1_6h[i]) or 
            np.isnan(support1_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price crosses above S1 with volume spike and above daily EMA50
            if (close[i] > support1_6h[i] and
                close[i] > ema_50_1d_6h[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price crosses below R1 with volume spike and below daily EMA50
            elif (close[i] < resistance1_6h[i] and
                  close[i] < ema_50_1d_6h[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below pivot or below daily EMA50
            if (close[i] < pivot_point_6h[i] or 
                close[i] < ema_50_1d_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above pivot or above daily EMA50
            if (close[i] > pivot_point_6h[i] or 
                close[i] > ema_50_1d_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Pivot_S1R1_EMA50_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-14 11:31
