#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian(20) breakout with 1-week trend filter and volume confirmation.
Long when price breaks above 20-period high and 1-week EMA50 rising with volume spike.
Short when price breaks below 20-period low and 1-week EMA50 falling with volume spike.
Exit when price returns to 10-period midpoint or 1-week EMA50 reverses.
Designed for low trade frequency by requiring breakout + trend + volume confluence.
Works in bull markets via breakouts and bear markets via breakdowns.
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
    
    # Donchian channels: 20-period high/low
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-period midpoint for exit
    mid_point = (high_max + low_min) / 2.0
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1-week close for trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(mid_point[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: break above 20-period high with 1w EMA50 rising and volume spike
            if close[i] > high_max[i] and ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-period low with 1w EMA50 falling and volume spike
            elif close[i] < low_min[i] and ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to midpoint or 1-week EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= midpoint or 1w EMA50 turns down
                if close[i] <= mid_point[i] or ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price >= midpoint or 1w EMA50 turns up
                if close[i] >= mid_point[i] or ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_DonchianBreakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0