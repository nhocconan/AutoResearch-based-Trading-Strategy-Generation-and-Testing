# Strategy: 4h_Camarilla_R1S1_VolumeSpike_ADX25

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.411 | +40.7% | -10.6% | 335 | KEEP |
| ETHUSDT | 0.049 | +21.6% | -13.6% | 315 | KEEP |
| SOLUSDT | 0.615 | +79.5% | -28.6% | 246 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.056 | -4.4% | -7.7% | 123 | DISCARD |
| ETHUSDT | 0.880 | +20.0% | -10.9% | 102 | KEEP |
| SOLUSDT | -0.356 | -0.3% | -13.4% | 87 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
4h Camarilla Pivot R1/S1 Breakout with Volume Spike and ADX Trend Filter
Long: Close breaks above R1 + volume > 2.0 x 4h volume MA(20) + ADX(14) > 25
Short: Close breaks below S1 + volume > 2.0 x 4h volume MA(20) + ADX(14) > 25
Exit: Close crosses back below R1 (long) or above S1 (short)
Uses 1D Camarilla levels for structure, volume for confirmation, ADX for trend filter
Target: 25-35 trades/year per symbol (100-140 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using previous day to avoid look-ahead
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    r1_values = r1.values
    s1_values = s1.values
    
    # Align to 4h timeframe (wait for 1D bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_values)
    
    # 4h volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) on 4h data
    # Calculate +DI, -DI, DX
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(high, dtype=float)
    plus_di = np.zeros_like(high, dtype=float)
    minus_di = np.zeros_like(high, dtype=float)
    dx = np.zeros_like(high, dtype=float)
    adx = np.zeros_like(high, dtype=float)
    
    # Initial values
    atr[period] = np.mean(tr[:period+1])
    plus_dm_smoothed = np.sum(plus_dm[:period+1])
    minus_dm_smoothed = np.sum(minus_dm[:period+1])
    
    for i in range(period + 1, len(high)):
        # Wilder's smoothing
        atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
        plus_dm_smoothed = plus_dm_smoothed - (plus_dm_smoothed / period) + plus_dm[i]
        minus_dm_smoothed = minus_dm_smoothed - (minus_dm_smoothed / period) + minus_dm[i]
        
        # Avoid division by zero
        if atr[i] != 0:
            plus_di[i] = (plus_dm_smoothed / atr[i]) * 100
            minus_di[i] = (minus_dm_smoothed / atr[i]) * 100
            dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        else:
            plus_di[i] = 0
            minus_di[i] = 0
            dx[i] = 0
    
    # Smoothed ADX
    adx[2*period] = np.mean(dx[period:2*period+1])
    for i in range(2*period + 1, len(high)):
        adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    # For indices before sufficient data, set to 0
    adx[:2*period] = 0
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(60, 2*period + 10)  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        adx_val = adx[i]
        
        if position == 0:
            # Long: break above R1 + volume spike + ADX > 25
            if price > r1_aligned[i] and vol > 2.0 * vol_ma_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + volume spike + ADX > 25
            elif price < s1_aligned[i] and vol > 2.0 * vol_ma_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close back below R1
            if price < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close back above S1
            if price > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_VolumeSpike_ADX25"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 23:19
