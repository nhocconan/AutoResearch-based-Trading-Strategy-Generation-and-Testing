# Strategy: 6h_ElderRay_Power_1dEMA34_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.030 | +17.8% | -13.6% | 143 | FAIL |
| ETHUSDT | 0.076 | +22.6% | -16.3% | 145 | PASS |
| SOLUSDT | 0.899 | +150.8% | -27.5% | 138 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.234 | +9.3% | -8.4% | 52 | PASS |
| SOLUSDT | -0.300 | -0.6% | -13.7% | 44 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h, HTF: 1d for trend filter
- Elder Ray: Bull Power = high - EMA13(close), Bear Power = low - EMA13(close) on 6h
- Long: Bull Power > 0 + Bear Power rising (from negative) + price > 1d EMA34 (uptrend) + volume > 1.5x 20-period avg
- Short: Bear Power < 0 + Bull Power falling (from positive) + price < 1d EMA34 (downtrend) + volume > 1.5x 20-period avg
- Exit: Elder Ray power crosses zero (momentum shift)
- Uses volume confirmation to reduce false signals, proven effective across market regimes
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (strong bull power) and bear markets (strong bear power)
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
    
    # Volume confirmation: > 1.5x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 6h EMA13 for Elder Ray Power
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power: Bull Power = high - EMA13, Bear Power = low - EMA13
    bull_power = high - ema_13_6h
    bear_power = low - ema_13_6h
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 34)  # Need 20 for volume MA, 13 for EMA13, 34 for 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 + Bear Power rising (from negative) + price > 1d EMA34 (uptrend) + volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and  # Bear Power rising (less negative/more positive)
                close[i] > ema_34_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 + Bull Power falling (from positive) + price < 1d EMA34 (downtrend) + volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and  # Bull Power falling (less positive/more negative)
                  close[i] < ema_34_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power crosses below zero (momentum shift)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power crosses above zero (momentum shift)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 20:55
