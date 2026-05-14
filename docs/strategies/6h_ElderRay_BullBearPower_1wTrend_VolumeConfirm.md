# Strategy: 6h_ElderRay_BullBearPower_1wTrend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.102 | +24.6% | -6.3% | 76 | PASS |
| ETHUSDT | 0.026 | +21.1% | -13.3% | 54 | PASS |
| SOLUSDT | 0.779 | +91.4% | -15.5% | 49 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.393 | +3.7% | -3.3% | 17 | FAIL |
| ETHUSDT | 0.551 | +12.1% | -6.0% | 18 | PASS |
| SOLUSDT | -0.697 | -1.4% | -8.2% | 15 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1wTrend_VolumeConfirm
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-week EMA trend filter and volume confirmation.
Elder Ray measures bull/bear power relative to EMA13. Long when Bear Power > 0 (bulls in control) with 1w uptrend and volume spike.
Short when Bull Power < 0 (bears in control) with 1w downtrend and volume spike.
Works in bull/bear regimes by using 1w trend to filter counter-trend whipsaws. Volume confirms institutional participation.
Target: 12-37 trades/year (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1-week EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily data for EMA13 (used in Elder Ray calculation)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on daily close
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align daily EMA13 to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34(1w) and EMA13(1d)
    start_idx = max(34, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bear Power > 0 (bulls in control) + 1w uptrend + volume spike
            long_setup = (bear_power[i] > 0) and (close[i] > ema_34_1w_aligned[i]) and volume_spike[i]
            # Short: Bull Power < 0 (bears in control) + 1w downtrend + volume spike
            short_setup = (bull_power[i] < 0) and (close[i] < ema_34_1w_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Bear Power <= 0 OR 1w trend turns down
            if (bear_power[i] <= 0) or (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bull Power >= 0 OR 1w trend turns up
            if (bull_power[i] >= 0) or (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 12:34
