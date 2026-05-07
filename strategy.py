#!/usr/bin/env python3
"""
1d_SwingPoint_Breakout_WeeklyTrend_Volume
Hypothesis: Daily swing point breakouts with weekly trend and volume confirmation capture institutional order flow. Works in bull/bear via trend filter and low trade frequency.
"""
name = "1d_SwingPoint_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate swing points (10-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    swing_high = high_series.rolling(window=10, center=False).max().values
    swing_low = low_series.rolling(window=10, center=False).min().values
    
    # Align swing points (already daily, no shift needed)
    swing_high_aligned = swing_high
    swing_low_aligned = swing_low
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need sufficient warmup for rolling
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above swing high + weekly uptrend + volume spike
            if (close[i] > swing_high_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below swing low + weekly downtrend + volume spike
            elif (close[i] < swing_low_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite swing level
            if position == 1:
                if close[i] <= swing_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= swing_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals