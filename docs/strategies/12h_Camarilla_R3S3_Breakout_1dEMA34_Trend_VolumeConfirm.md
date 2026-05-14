# Strategy: 12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.214 | +14.5% | -5.8% | 98 | FAIL |
| ETHUSDT | 0.084 | +23.9% | -4.9% | 85 | PASS |
| SOLUSDT | -0.100 | +10.7% | -20.4% | 94 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.002 | +19.3% | -4.7% | 33 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
- Uses Camarilla pivot levels from 12h timeframe for precise entry/exit points
- 1d EMA34 defines higher timeframe trend: only trade breakouts in trend direction
- Volume confirmation (> 1.8x 20-period average) filters false breakouts
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend
- Camarilla R3/S3 levels provide institutional-grade support/resistance with proven edge
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
    
    # Calculate 12h Camarilla pivot levels (R3, S3)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla formula: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3_12h = close_12h + (high_12h - low_12h) * 1.1 / 4
    camarilla_s3_12h = close_12h - (high_12h - low_12h) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 with 1d uptrend and volume
            long_breakout = (close[i] > camarilla_r3_aligned[i] and 
                           close[i] > ema_34_1d_aligned[i] and
                           volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: price breaks below Camarilla S3 with 1d downtrend and volume
            short_breakout = (close[i] < camarilla_s3_aligned[i] and 
                            close[i] < ema_34_1d_aligned[i] and
                            volume[i] > 1.8 * vol_ma[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S3 or 1d trend turns bearish
                if (close[i] < camarilla_s3_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Camarilla R3 or 1d trend turns bullish
                if (close[i] > camarilla_r3_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 16:52
