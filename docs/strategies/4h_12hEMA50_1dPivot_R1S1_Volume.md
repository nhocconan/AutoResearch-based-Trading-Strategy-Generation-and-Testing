# Strategy: 4h_12hEMA50_1dPivot_R1S1_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.064 | +23.0% | -11.3% | 199 | PASS |
| ETHUSDT | 0.121 | +25.7% | -12.4% | 181 | PASS |
| SOLUSDT | 0.675 | +87.8% | -27.4% | 157 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.376 | -7.0% | -8.4% | 75 | FAIL |
| ETHUSDT | 0.654 | +15.7% | -7.6% | 66 | PASS |
| SOLUSDT | -0.643 | -4.2% | -13.3% | 54 | FAIL |

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
    
    # Load 12h data for trend context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
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
            np.isnan(ema_50_12h_aligned[i])):
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
            # Long: Price breaks above R1 with volume spike AND above 12h EMA50
            if (close[i] > resistance1_4h[i] and 
                volume_ratio > vol_threshold and
                close[i] > ema_50_12h_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S1 with volume spike AND below 12h EMA50
            elif (close[i] < support1_4h[i] and 
                  volume_ratio > vol_threshold and
                  close[i] < ema_50_12h_aligned[i]):
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

name = "4h_12hEMA50_1dPivot_R1S1_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-14 11:42
