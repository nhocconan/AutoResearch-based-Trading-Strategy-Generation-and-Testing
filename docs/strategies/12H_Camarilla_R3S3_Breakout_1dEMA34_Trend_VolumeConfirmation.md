# Strategy: 12H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.148 | +26.2% | -5.3% | 73 | PASS |
| ETHUSDT | 0.161 | +27.4% | -7.0% | 64 | PASS |
| SOLUSDT | 0.167 | +29.1% | -19.7% | 66 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.035 | -2.3% | -7.7% | 33 | FAIL |
| ETHUSDT | 0.764 | +15.6% | -4.8% | 24 | PASS |
| SOLUSDT | -0.247 | +2.4% | -12.1% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years). Uses discrete position sizing (0.30) to minimize fee churn.
Works in both bull/bear via 1d trend filter and volume confirmation to avoid false breakouts.
Camarilla R3/S3 levels provide balanced breakout signals with moderate whipsaw reduction.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Camarilla levels (using previous 12h bar's range)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla R3, S3 levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    range_12h = high_12h - low_12h
    camarilla_r3_12h = close_12h + range_12h * 1.1 / 4
    camarilla_s3_12h = close_12h - range_12h * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need EMA34, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA34 = uptrend, close < 1d EMA34 = downtrend
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 12h volume > 2.0x 20-period MA (stricter to reduce trades)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND uptrend AND volume confirmation
            if close[i] > camarilla_r3_aligned[i] and trend_up and vol_filter:
                signals[i] = 0.30
                position = 1
            # Short: Break below Camarilla S3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and trend_down and vol_filter:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: reverse signal or break of opposite Camarilla level
            exit_signal = False
            if position == 1:
                # Exit long on break below Camarilla S3
                if close[i] < camarilla_s3_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Camarilla R3
                if close[i] > camarilla_r3_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 14:43
