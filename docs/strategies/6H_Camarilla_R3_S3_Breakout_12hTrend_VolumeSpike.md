# Strategy: 6H_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.549 | +41.1% | -7.1% | 188 | PASS |
| ETHUSDT | 0.262 | +32.0% | -10.4% | 163 | PASS |
| SOLUSDT | 0.718 | +84.1% | -19.3% | 139 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.260 | -4.1% | -5.5% | 76 | FAIL |
| ETHUSDT | 1.518 | +28.3% | -5.9% | 65 | PASS |
| SOLUSDT | -0.174 | +3.5% | -8.6% | 56 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6H_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
# Hypothesis: 6-hour timeframe with 12-hour trend filter and volume spike confirmation. 
# Uses Camarilla R3/S3 levels from daily pivot calculation for institutional breakout/breakdown levels.
# Works in bull markets (breakouts continue with trend) and bear markets (mean reversion from extremes via short entries).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6H_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # Volume confirmation: volume > 2.0 * 24-period average (more stringent for 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d data for Camarilla R3/S3 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R3 = Close + 1.1*(High-Low)*1.1/4, S3 = Close - 1.1*(High-Low)*1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    rang = prev_high - prev_low
    R3 = prev_close + 1.1 * rang * 1.1 / 4
    S3 = prev_close - 1.1 * rang * 1.1 / 4
    
    # Align daily Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA50
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + price above 12h EMA50 (uptrend)
            if (close[i] > R3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + price below 12h EMA50 (downtrend)
            elif (close[i] < S3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Camarilla H-L range (between S3 and R3) OR closes below 12h EMA50
            if (close[i] < R3_aligned[i] and close[i] > S3_aligned[i]) or \
               close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Camarilla H-L range (between S3 and R3) OR closes above 12h EMA50
            if (close[i] < R3_aligned[i] and close[i] > S3_aligned[i]) or \
               close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 10:43
