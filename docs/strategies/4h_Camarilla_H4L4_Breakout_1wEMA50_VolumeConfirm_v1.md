# Strategy: 4h_Camarilla_H4L4_Breakout_1wEMA50_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.250 | +28.6% | -6.2% | 139 | PASS |
| ETHUSDT | 0.117 | +25.0% | -9.9% | 120 | PASS |
| SOLUSDT | 0.041 | +21.7% | -9.2% | 89 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.968 | -4.1% | -5.9% | 50 | FAIL |
| ETHUSDT | 0.113 | +7.0% | -8.9% | 49 | PASS |
| SOLUSDT | -0.093 | +4.9% | -7.3% | 36 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 1w EMA50 trend filter and volume spike confirmation.
- H4/L4 are stronger Camarilla levels (1.5/2.0 multiples) requiring stronger momentum to break.
- 1w EMA50 ensures we trade only in the direction of the weekly trend, reducing whipsaws in choppy markets.
- Volume spike (>2.0x 20-bar average) confirms institutional participation in breakouts.
- Position size 0.25 balances profit potential and drawdown control.
- Target trades: 60-120 total over 4 years (15-30/year) to minimize fee drag.
- Works in bull/bear markets via weekly trend filter and high-probability breakout logic.
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
    
    # Get 1w data ONCE before loop for EMA filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels from prior 4h bar
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Camarilla formulas
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    h4 = pivot + (range_hl * 1.5 / 2)  # H4 level
    l4 = pivot - (range_hl * 1.5 / 2)  # L4 level
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1  # Need enough for EMA and Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(h4[i]) or np.isnan(l4[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Long breakout: price above H4 AND above 1w EMA50
                if close[i] > h4[i] and close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below L4 AND below 1w EMA50
                elif close[i] < l4[i] and close[i] < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L4 OR crosses below 1w EMA50
            if close[i] < l4[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H4 OR crosses above 1w EMA50
            if close[i] > h4[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 01:43
