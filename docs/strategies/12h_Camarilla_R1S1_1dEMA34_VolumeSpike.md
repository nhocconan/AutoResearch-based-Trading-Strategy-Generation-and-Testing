# Strategy: 12h_Camarilla_R1S1_1dEMA34_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.554 | +1.8% | -16.5% | 172 | DISCARD |
| ETHUSDT | 0.014 | +20.6% | -15.3% | 149 | KEEP |
| SOLUSDT | -0.040 | +12.6% | -30.9% | 153 | DISCARD |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.286 | +9.5% | -6.7% | 54 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
- Camarilla levels calculated from prior 1d bar's high-low-close
- Long: Close breaks above R1 (moderate volatility expansion) + price > 1d EMA34 (uptrend) + volume > 1.5x 20-period avg
- Short: Close breaks below S1 (moderate volatility expansion) + price < 1d EMA34 (downtrend) + volume > 1.5x 20-period avg
- Exit: Close reverts to mean (returns to Camarilla pivot point) OR opposite breakout
- 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe
- Works in both bull (trend continuation via breakouts) and bear (mean reversion via pivot returns)
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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Camarilla calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R1 = Pivot + Range * 1.1/12, S1 = Pivot - Range * 1.1/12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = pivot_1d + range_1d * 1.1 / 12.0
    s1_1d = pivot_1d - range_1d * 1.1 / 12.0
    pp_1d = pivot_1d  # Pivot point for mean reversion exit
    
    # Align Camarilla levels to 12h timeframe (available after 1d bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for volume MA, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above R1 + price > 1d EMA34 (uptrend) + volume spike
            if volume_spike and close[i] > r1_aligned[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 + price < 1d EMA34 (downtrend) + volume spike
            elif volume_spike and close[i] < s1_aligned[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close returns to pivot point (mean reversion) OR breaks below S1 (reversal)
            if close[i] <= pp_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close returns to pivot point (mean reversion) OR breaks above R1 (reversal)
            if close[i] >= pp_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-05-04 12:02
