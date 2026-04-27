#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel (20-period high) and 12h EMA50 is rising.
# Short when price breaks below Donchian lower channel (20-period low) and 12h EMA50 is falling.
# Volume confirmation requires current volume > 1.5x 20-period average to ensure institutional participation.
# Designed for low trade frequency (target: 75-200 total trades over 4 years) to minimize fee drag.
# Works in bull markets (captures uptrend breakouts) and bear markets (captures downtrend breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12-hour EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian upper, uptrend, volume
        if (close[i] > high_max[i] and 
            close[i] > ema50_12h_aligned[i] and  # price above EMA50 (uptrend)
            volume_filter[i]):
            signals[i] = 0.30
            position = 1
        # Short condition: price breaks below Donchian lower, downtrend, volume
        elif (close[i] < low_min[i] and
              close[i] < ema50_12h_aligned[i] and  # price below EMA50 (downtrend)
              volume_filter[i]):
            signals[i] = -0.30
            position = -1
        # Exit conditions: trend reversal or price retracement to EMA50
        elif position == 1 and (close[i] <= ema50_12h_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] >= ema50_12h_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0