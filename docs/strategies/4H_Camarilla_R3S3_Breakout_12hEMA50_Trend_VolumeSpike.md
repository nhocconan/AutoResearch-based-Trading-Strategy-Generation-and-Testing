# Strategy: 4H_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.093 | +24.2% | -9.1% | 267 | PASS |
| ETHUSDT | 0.167 | +29.0% | -15.9% | 258 | PASS |
| SOLUSDT | 0.958 | +158.2% | -22.5% | 244 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.530 | -0.1% | -7.2% | 90 | FAIL |
| ETHUSDT | 0.555 | +15.4% | -12.2% | 87 | PASS |
| SOLUSDT | 0.153 | +7.8% | -11.5% | 89 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 Breakout + 12h EMA50 Trend + Volume Spike
Camarilla pivot levels (R3/S3) act as strong intraday support/resistance. 
Breakout above R3 or below S3 with 12h EMA50 trend alignment and volume confirmation
captures sustained momentum moves. 4h timeframe balances noise reduction and trade frequency.
Target: 25-50 trades/year (100-200 over 4 years) with discrete sizing 0.25.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d[0] = df_1d['close'].iloc[0]
    prev_high_1d[0] = df_1d['high'].iloc[0]
    prev_low_1d[0] = df_1d['low'].iloc[0]
    
    camarilla_range = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close_1d - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50_12h, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R3 (breakout resistance) AND price > 12h EMA50 (uptrend) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < S3 (breakdown support) AND price < 12h EMA50 (downtrend) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close back inside previous day's Camarilla H-L range OR loss of trend
            exit_signal = False
            if position == 1:
                # Exit long when close < S3 (breakdown of support) OR price < 12h EMA50
                if close[i] < s3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > R3 (breakout of resistance) OR price > 12h EMA50
                if close[i] > r3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 15:38
