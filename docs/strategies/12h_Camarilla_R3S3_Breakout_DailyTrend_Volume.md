# Strategy: 12h_Camarilla_R3S3_Breakout_DailyTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.189 | +27.8% | -6.6% | 69 | PASS |
| ETHUSDT | 0.083 | +23.7% | -8.9% | 61 | PASS |
| SOLUSDT | 0.165 | +28.9% | -22.3% | 60 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.564 | +1.2% | -6.9% | 24 | FAIL |
| ETHUSDT | 0.322 | +10.1% | -6.1% | 25 | PASS |
| SOLUSDT | -0.119 | +3.8% | -9.2% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h Camarilla R3/S3 Breakout + Volume Spike + Daily Trend Filter
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance. Breaking these
levels with volume confirmation and aligned with daily trend (price > EMA34) captures
strong moves in both bull and bear markets. Designed for low trade frequency (~20/year)
to minimize fee decay while capturing sustained trends.
"""
name = "12h_Camarilla_R3S3_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # === Camarilla Pivot Levels from previous day ===
    # Using high, low, close from previous day to calculate today's levels
    # We'll calculate these on daily data and align to 12h
    daily_high = pd.Series(high).rolling(window=24, min_periods=24).max().values  # Approximate daily high from 12h bars (2 per day)
    daily_low = pd.Series(low).rolling(window=24, min_periods=24).min().values    # Approximate daily low
    daily_close = pd.Series(close).rolling(window=24, min_periods=24).last().values  # Last close in window as daily close
    
    # For more accurate daily data, use HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily data
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We use R3 and S3 as key levels
    daily_high_1d = df_1d['high'].values
    daily_low_1d = df_1d['low'].values
    daily_close_1d = df_1d['close'].values
    
    # Calculate for each day, then shift by 1 to use previous day's levels
    rng = daily_high_1d - daily_low_1d
    R3 = daily_close_1d + rng * 1.1 / 4
    S3 = daily_close_1d - rng * 1.1 / 4
    
    # Shift to get previous day's levels (today's R3/S3 based on yesterday's HLC)
    R3_prev = np.roll(R3, 1)
    S3_prev = np.roll(S3, 1)
    # First day has no previous, set to NaN
    R3_prev[0] = np.nan
    S3_prev[0] = np.nan
    
    # Align to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3_prev)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3_prev)
    
    # === Daily EMA34 for trend filter ===
    ema_34_1d = pd.Series(daily_close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume Spike (24-period on 12h) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + price above daily EMA34 (uptrend)
            if (close[i] > R3_12h[i] and 
                vol_spike[i] and
                close[i] > ema_34_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + price below daily EMA34 (downtrend)
            elif (close[i] < S3_12h[i] and 
                  vol_spike[i] and
                  close[i] < ema_34_12h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (reversal) OR volume spike fails
            if close[i] < S3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 (reversal) OR volume spike fails
            if close[i] > R3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 06:42
