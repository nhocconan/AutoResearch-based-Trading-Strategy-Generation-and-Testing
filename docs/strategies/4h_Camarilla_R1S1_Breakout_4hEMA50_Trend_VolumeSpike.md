# Strategy: 4h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.400 | +3.9% | -15.6% | 346 | DISCARD |
| ETHUSDT | 0.297 | +36.9% | -16.8% | 304 | KEEP |
| SOLUSDT | 0.626 | +80.2% | -29.0% | 245 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.203 | +8.5% | -10.9% | 115 | KEEP |
| SOLUSDT | -0.557 | -3.9% | -14.6% | 86 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and volume confirmation (>2.0x 20-bar MA). 
4h timeframe balances trade frequency and signal quality. Camarilla R1/S1 provides intraday support/resistance. 
4h EMA50 filter ensures trading with intermediate trend. Volume confirmation adds conviction. 
Discrete sizing 0.28 balances profit and fee drag. Target: 30-60 trades/year (~120-240 over 4 years) to stay within fee drag limits.
Works in bull/bear: trend filter adapts to market direction, volume confirms breakout validity, and tight stops prevent large drawdowns.
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
    
    # Get 4h data for trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar's OHLC
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_close_4h = df_4h['close'].shift(1).values
    
    camarilla_range = prev_high_4h - prev_low_4h
    r1 = prev_close_4h + 1.1 * camarilla_range / 12   # R1 level
    s1 = prev_close_4h - 1.1 * camarilla_range / 12   # S1 level
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 4h EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND 4h trend bullish (close > EMA50) AND volume confirm
            long_setup = (close[i] > r1_aligned[i]) and \
                         (close[i] > ema_50_4h_aligned[i]) and \
                         volume_confirm[i]
            # Short: price breaks below S1 AND 4h trend bearish (close < EMA50) AND volume confirm
            short_setup = (close[i] < s1_aligned[i]) and \
                          (close[i] < ema_50_4h_aligned[i]) and \
                          volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.28
                position = 1
            elif short_setup:
                signals[i] = -0.28
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.28
            # Exit: price re-enters Camarilla R1/S1 range OR 4h trend turns bearish
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or \
               (close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.28
            # Exit: price re-enters Camarilla R1/S1 range OR 4h trend turns bullish
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or \
               (close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 14:08
