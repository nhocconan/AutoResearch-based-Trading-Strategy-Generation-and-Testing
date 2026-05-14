# Strategy: 4h_Vortex_Trend_Filter_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.124 | +10.8% | -16.5% | 339 | FAIL |
| ETHUSDT | 0.290 | +40.0% | -17.8% | 347 | PASS |
| SOLUSDT | 1.049 | +209.7% | -24.8% | 298 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.028 | +5.3% | -9.2% | 110 | PASS |
| SOLUSDT | -0.241 | -0.7% | -16.0% | 108 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Vortex_Trend_Filter_Volume
Hypothesis: Vortex Indicator (VI) captures trend direction while filtering noise. 
Long when VI+ > VI- and VI+ rising, short when VI- > VI+ and VI- rising, 
with volume confirmation and EMA50 trend filter. 
Designed for low trade frequency (<30/year) to avoid fee drag in choppy markets.
"""

name = "4h_Vortex_Trend_Filter_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Vortex Indicator: VM+ and VM-
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0.0
    vm_minus[0] = 0.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (EMA-like)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    vm_plus_sum = wilders_smoothing(vm_plus, period)
    vm_minus_sum = wilders_smoothing(vm_minus, period)
    tr_sum = wilders_smoothing(tr, period)
    
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Trend filter: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: VI+ > VI- and VI+ rising, uptrend, volume confirmation
            if vi_plus[i] > vi_minus[i] and vi_plus[i] > vi_plus[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ and VI- rising, downtrend, volume confirmation
            elif vi_minus[i] > vi_plus[i] and vi_minus[i] > vi_minus[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- crosses above VI+ or trend fails
            if vi_minus[i] > vi_plus[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ crosses above VI- or trend fails
            if vi_plus[i] > vi_minus[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 07:23
