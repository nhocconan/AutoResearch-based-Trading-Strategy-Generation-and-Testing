#!/usr/bin/env python3

# 1d_1w_donchian_breakout_with_volume_confirmation
# Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
# Works in bull/bear by using weekly trend to filter breakout direction and volume to avoid false signals.
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.

name = "1d_1w_donchian_breakout_with_volume_confirmation"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA for trend filter (21-period)
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.3x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above Donchian high with weekly uptrend and volume
        if (close[i] > high_20[i] and 
            close[i] > ema_21_1w_aligned[i] and 
            vol_confirm[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below Donchian low with weekly downtrend and volume
        elif (close[i] < low_20[i] and 
              close[i] < ema_21_1w_aligned[i] and 
              vol_confirm[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or price returns to opposite Donchian level
        elif position == 1 and close[i] < low_20[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > high_20[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals