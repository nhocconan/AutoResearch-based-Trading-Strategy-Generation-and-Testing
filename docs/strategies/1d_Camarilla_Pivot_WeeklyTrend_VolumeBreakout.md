# Strategy: 1d_Camarilla_Pivot_WeeklyTrend_VolumeBreakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.068 | +22.3% | -18.0% | 23 | PASS |
| ETHUSDT | -0.147 | +7.5% | -17.7% | 19 | FAIL |
| SOLUSDT | -0.376 | -26.7% | -56.3% | 41 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.711 | +16.3% | -7.7% | 6 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_WeeklyTrend_VolumeBreakout
Hypothesis: On daily timeframe, use Camarilla pivot levels (R1/S1) as support/resistance. Enter long when price breaks above R1 with volume surge and weekly uptrend (EMA8 > EMA21), short when price breaks below S1 with volume surge and weekly downtrend. Exit on opposite Camarilla level break with volume. Weekly trend filter avoids counter-trend trades during extended trends, while Camarilla levels provide institutional reference points. Volume surge confirms institutional participation. Designed for low trade frequency (~10-25/year) to minimize fee decay in both bull and bear markets.
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 21:
        return np.zeros(n)
    
    # Calculate weekly 8 and 21 EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema8_weekly = pd.Series(close_weekly).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema8_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema8_weekly)
    ema21_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema21_weekly)
    
    # Weekly trend: bullish when EMA8 > EMA21
    weekly_uptrend = ema8_weekly_aligned > ema21_weekly_aligned
    weekly_downtrend = ema8_weekly_aligned < ema21_weekly_aligned
    
    # Previous day's data for Camarilla calculation
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First day uses current day's values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Calculate Camarilla pivot levels
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Volume confirmation: current volume > 2.0x 50-day average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_surge = volume > (vol_ma_50 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema8_weekly_aligned[i]) or np.isnan(ema21_weekly_aligned[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly trend alignment and volume surge
        long_entry = close[i] > r1[i] and weekly_uptrend[i] and volume_surge[i]
        short_entry = close[i] < s1[i] and weekly_downtrend[i] and volume_surge[i]
        
        # Exit on opposite Camarilla level break with volume surge
        long_exit = close[i] < s1[i] and volume_surge[i]
        short_exit = close[i] > r1[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_Pivot_WeeklyTrend_VolumeBreakout"
timeframe = "1d"
leverage = 1.0
```

## Last Updated
2026-04-28 04:27
