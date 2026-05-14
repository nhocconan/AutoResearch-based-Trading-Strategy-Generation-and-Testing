# Strategy: 12H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.171 | +27.1% | -5.3% | 72 | PASS |
| ETHUSDT | 0.145 | +26.6% | -7.0% | 62 | PASS |
| SOLUSDT | 0.157 | +28.3% | -19.7% | 65 | PASS |

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
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Uses Camarilla pivot levels from 12h timeframe for entry/exit, combined with
1d EMA34 trend filter to avoid counter-trend trades. Volume spike confirms breakout momentum.
Designed for 12h timeframe to capture intermediate-term moves with lower trade frequency.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
Uses discrete position sizing (0.30) to balance return and fee drag.
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
    
    # Calculate 12h EMA50 for trend filter (secondary confirmation)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d EMA34 for primary trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Camarilla pivot levels (R3, S3, R4, S4)
    # Based on previous 12h bar's OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    
    # Typical price for Camarilla calculation
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    camarilla_r3 = close_12h + (range_12h * 1.1 / 4)
    camarilla_s3 = close_12h - (range_12h * 1.1 / 4)
    camarilla_r4 = close_12h + (range_12h * 1.1 / 2)
    camarilla_s4 = close_12h - (range_12h * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (previous 12h bar values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filters: both 12h EMA50 and 1d EMA34 must agree
        trend_12h_up = close[i] > ema_50_12h_aligned[i]
        trend_12h_down = close[i] < ema_50_12h_aligned[i]
        trend_1d_up = close[i] > ema_34_1d_aligned[i]
        trend_1d_down = close[i] < ema_34_1d_aligned[i]
        
        trend_up = trend_12h_up and trend_1d_up
        trend_down = trend_12h_down and trend_1d_down
        
        if position == 0:
            # Long: Break above Camarilla R3 AND uptrend on both timeframes AND volume spike
            if close[i] > camarilla_r3_aligned[i] and trend_up and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: Break below Camarilla S3 AND downtrend on both timeframes AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and trend_down and volume_spike[i]:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: break of opposite Camarilla level (S3 for longs, R3 for shorts)
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

name = "12H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 15:14
