# Strategy: 6h_ElderRay_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.213 | +11.5% | -13.9% | 97 | FAIL |
| ETHUSDT | 0.200 | +30.3% | -10.8% | 97 | PASS |
| SOLUSDT | 1.244 | +186.5% | -13.2% | 82 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.506 | +14.3% | -9.6% | 39 | PASS |
| SOLUSDT | -0.468 | -0.2% | -10.3% | 21 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (measures buying/selling pressure)
- Long: Bull Power > 0 AND Bear Power rising (less negative) AND price > 12h EMA50 AND volume > 1.5x 20-period avg
- Short: Bear Power < 0 AND Bull Power falling (less positive) AND price < 12h EMA50 AND volume > 1.5x 20-period avg
- Exit: Elder Ray signals reverse OR price crosses 12h EMA50
- Works in bull (buy strength on dips) and bear (sell weakness on rallies)
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Buying pressure
    bear_power = low - ema_13   # Selling pressure (negative values)
    
    # Smoothed Elder Ray for trend confirmation (3-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Calculate 12h EMA50 for trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 20)  # Need 50 for safety, 13 for EMA13, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bull_power_smooth[i]) or
            np.isnan(bear_power_smooth[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Elder Ray signals
        bull_strong = bull_power_smooth[i] > 0 and bull_power_smooth[i] > bull_power_smooth[i-1]
        bear_weak = bear_power_smooth[i] < 0 and bear_power_smooth[i] > bear_power_smooth[i-1]  # Less negative
        bear_strong = bear_power_smooth[i] < 0 and bear_power_smooth[i] < bear_power_smooth[i-1]
        bull_weak = bull_power_smooth[i] > 0 and bull_power_smooth[i] < bull_power_smooth[i-1]  # Less positive
        
        if position == 0:
            # Long: Bull Power positive AND rising, Bear Power weak AND rising (less negative), price > 12h EMA50
            if (bull_strong and bear_weak and 
                volume_confirm and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND falling, Bull Power weak AND falling (less positive), price < 12h EMA50
            elif (bear_strong and bull_weak and 
                  volume_confirm and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR price < 12h EMA50 (trend flip)
            if bull_power_smooth[i] <= 0 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive OR price > 12h EMA50 (trend flip)
            if bear_power_smooth[i] >= 0 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 22:16
