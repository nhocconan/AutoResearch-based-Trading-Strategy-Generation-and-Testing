# Strategy: 6h_Pivot_R1_S1_Breakout_Volume_ATRFilter_V1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.453 | -10.6% | -36.6% | 2030 | FAIL |
| ETHUSDT | 0.035 | +17.1% | -25.0% | 2065 | PASS |
| SOLUSDT | 0.751 | +146.0% | -26.9% | 2028 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.681 | +21.3% | -13.3% | 729 | PASS |
| SOLUSDT | -0.326 | -4.7% | -25.4% | 705 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Pivot_R1_S1_Breakout_Volume_ATRFilter_V1
6h strategy using daily Camarilla pivot levels (R1/S1) with volume spike and ATR filter.
- Long: Price breaks above R1 + volume > 2x average + ATR > ATR(10) MA
- Short: Price breaks below S1 + volume > 2x average + ATR > ATR(10) MA
- Exit: Opposite signal or price crosses daily EMA34
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for regime filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    atr_filter = atr > atr_ma
    
    # Volume spike filter (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need 20 for volume MA + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: daily EMA34
        bull_regime = close[i] > ema_34_1d_aligned[i]
        bear_regime = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakdown_down = close[i] < s1_aligned[i]
        
        # Volume spike filter
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        # ATR filter
        atr_ok = atr_filter[i]
        
        if position == 0:
            # Long: bull regime + breakout above R1 + volume spike + ATR filter
            if bull_regime and breakout_up and volume_spike and atr_ok:
                signals[i] = 0.25
                position = 1
            # Short: bear regime + breakdown below S1 + volume spike + ATR filter
            elif bear_regime and breakdown_down and volume_spike and atr_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: regime change or price breaks below S1
            if not bull_regime or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: regime change or price breaks above R1
            if not bear_regime or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-18 13:26
