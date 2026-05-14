# Strategy: 4h_Camarilla_R1S1_Breakout_12hTrendFilter_VolumeSpike_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.428 | +36.6% | -7.1% | 284 | PASS |
| ETHUSDT | 0.400 | +37.3% | -9.8% | 261 | PASS |
| SOLUSDT | 0.287 | +37.3% | -17.4% | 225 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.347 | -3.5% | -5.6% | 113 | FAIL |
| ETHUSDT | 1.042 | +18.9% | -5.2% | 101 | PASS |
| SOLUSDT | 0.565 | +12.3% | -5.5% | 82 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrendFilter_VolumeSpike_v4
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 12h EMA50 trend filter and volume spike confirmation. Uses tighter entry (R1/S1) to reduce trade count and improve selectivity. Volume spike confirms breakout strength. Discrete sizing (0.25) to limit fee drag. Target: 20-40 trades/year per symbol for better generalization. Uses 12h HTF for more stable trend filter vs 1d to avoid whipsaw in sideways markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter and Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: R1/S1 (using close of previous 12h bar)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_12h + 1.1 * (high_12h - low_12h) / 12
    camarilla_s1 = close_12h - 1.1 * (high_12h - low_12h) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above EMA50)
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above R1 + 12h uptrend + volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and htf_12h_bullish and volume_spike[i]
            
            # Short setup: price breaks below S1 + 12h downtrend + volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and htf_12h_bearish and volume_spike[i]
            
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
            # Exit: price touches S1 (stop) OR 12h trend turns bearish
            if (close[i] <= camarilla_s1_aligned[i]) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches R1 (stop) OR 12h trend turns bullish
            if (close[i] >= camarilla_r1_aligned[i]) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrendFilter_VolumeSpike_v4"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 15:19
