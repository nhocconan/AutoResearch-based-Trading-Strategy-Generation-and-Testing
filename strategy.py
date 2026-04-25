#!/usr/bin/env python3
"""
6h Donchian(20) breakout + weekly pivot direction + volume confirmation
Hypothesis: Donchian channel breakouts capture momentum, while weekly pivot direction (from 1w timeframe) filters for higher-probability trades aligned with the dominant trend. Volume confirmation ensures institutional participation. Works in both bull/bear markets: in uptrends, take long breakouts above weekly pivot; in downtrends, take short breakouts below weekly pivot. Uses discrete sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    highest_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (wait for completed 1d bar)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Get 1w data for weekly pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot point (standard: (H+L+C)/3)
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    weekly_pivot_values = weekly_pivot.values
    
    # Align weekly pivot to 6h timeframe (wait for completed 1w bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_values)
    
    # Calculate ATR for volatility filter (14-period on 1d)
    tr1 = df_1d['high'][1:] - df_1d['low'][1:]
    tr2 = np.abs(df_1d['high'][1:] - df_1d['close'][:-1])
    tr3 = np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20) and ATR (14)
    start_idx = 35  # 20 for Donchian + 14 for ATR + 1 buffer
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = highest_20_aligned[i]
        donchian_low = lowest_20_aligned[i]
        weekly_pivot = weekly_pivot_aligned[i]
        atr_value = atr_1d_aligned[i]
        
        # Volume spike: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 1.5 * vol_ma_20
        
        # Breakout conditions: price breaks above/below Donchian channels
        bullish_breakout = curr_close > donchian_high
        bearish_breakout = curr_close < donchian_low
        
        # Exit conditions: opposite breakout or volatility contraction
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish breakout below Donchian low
                if bearish_breakout:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish breakout above Donchian high
                if bullish_breakout:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Donchian breakout + weekly pivot direction + volume spike
        if position == 0:
            # Long: break above Donchian high AND price above weekly pivot (uptrend bias)
            long_condition = bullish_breakout and (curr_close > weekly_pivot) and volume_spike
            # Short: break below Donchian low AND price below weekly pivot (downtrend bias)
            short_condition = bearish_breakout and (curr_close < weekly_pivot) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0