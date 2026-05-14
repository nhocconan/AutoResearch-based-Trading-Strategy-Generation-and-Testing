# Strategy: 4h_12hATR_Vol_Filter_Pivot_Breakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.183 | +11.2% | -14.1% | 247 | FAIL |
| ETHUSDT | 0.220 | +32.0% | -12.0% | 216 | PASS |
| SOLUSDT | 0.550 | +73.5% | -28.4% | 193 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.043 | +6.0% | -8.4% | 83 | PASS |
| SOLUSDT | -0.316 | -0.1% | -13.5% | 69 | FAIL |

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
    
    # Load 12h data for ATR-based volatility filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range and ATR(14) on 12h
    tr = np.zeros(len(df_12h))
    tr[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(df_12h)):
        tr[i] = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
    
    atr_12h = np.zeros(len(df_12h))
    atr_12h[:13] = np.nan
    if len(df_12h) >= 14:
        atr_12h[13] = np.mean(tr[:14])
        for i in range(14, len(df_12h)):
            atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
    
    # Align 12h ATR to 4h timeframe
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Load 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (S1, S2, R1, R2)
    pivot_point = np.full_like(close_1d, np.nan)
    resistance1 = np.full_like(close_1d, np.nan)
    resistance2 = np.full_like(close_1d, np.nan)
    support1 = np.full_like(close_1d, np.nan)
    support2 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 2:
        for i in range(1, len(close_1d)):
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            
            pp = (ph + pl + pc) / 3.0
            r1 = 2 * pp - pl
            r2 = pp + (ph - pl)
            s1 = 2 * pp - ph
            s2 = pp - (ph - pl)
            
            pivot_point[i] = pp
            resistance1[i] = r1
            resistance2[i] = r2
            support1[i] = s1
            support2[i] = s2
    
    # Align 1d indicators to 4h timeframe
    pivot_point_4h = align_htf_to_ltf(prices, df_1d, pivot_point)
    resistance1_4h = align_htf_to_ltf(prices, df_1d, resistance1)
    resistance2_4h = align_htf_to_ltf(prices, df_1d, resistance2)
    support1_4h = align_htf_to_ltf(prices, df_1d, support1)
    support2_4h = align_htf_to_ltf(prices, df_1d, support2)
    
    # Volume spike detection (20-period average)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_point_4h[i]) or 
            np.isnan(resistance1_4h[i]) or
            np.isnan(resistance2_4h[i]) or
            np.isnan(support1_4h[i]) or
            np.isnan(support2_4h[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_12h_aligned[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike and adequate volatility
            if (close[i] > resistance1_4h[i] and 
                volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S1 with volume spike and adequate volatility
            elif (close[i] < support1_4h[i] and 
                  volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price closes below pivot point (mean reversion)
            if close[i] < pivot_point_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price closes above pivot point (mean reversion)
            if close[i] > pivot_point_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12hATR_Vol_Filter_Pivot_Breakout"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-14 11:45
