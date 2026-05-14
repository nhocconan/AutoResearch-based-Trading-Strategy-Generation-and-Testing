# Strategy: 12h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.310 | +12.8% | -7.2% | 74 | FAIL |
| ETHUSDT | 0.052 | +22.7% | -4.7% | 61 | PASS |
| SOLUSDT | 0.033 | +19.9% | -26.2% | 61 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.087 | +6.7% | -3.6% | 29 | PASS |
| SOLUSDT | -0.774 | -2.2% | -11.1% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 Breakout with 1d EMA50 Trend Filter and Volume Spike
- Uses Camarilla pivot levels (R3/S3) from daily timeframe for structure-based entries
- 1d EMA50 defines higher timeframe trend filter: only trade in direction of 1d trend
- Volume confirmation (> 2.0x 20-period average) filters weak signals
- Exit when price retouches Camarilla pivot point (PP) or trend reverses
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by combining mean reversion at extremes with trend filter
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
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations: based on previous day's range
    PP = (high_1d + low_1d + close_1d) / 3
    R = high_1d - low_1d
    R3 = PP + R * 1.1 / 4
    S3 = PP - R * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above 1d EMA50 AND volume spike
            if (close[i] > R3_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below 1d EMA50 AND volume spike
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price retouches PP OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when price retouches PP OR closes below 1d EMA50
                if (close[i] <= PP_aligned[i] or close[i] < ema_50_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price retouches PP OR closes above 1d EMA50
                if (close[i] >= PP_aligned[i] or close[i] > ema_50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 16:37
