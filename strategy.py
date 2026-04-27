#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirm
Hypothesis: 6h Donchian(20) breakouts filtered by weekly pivot direction (from 1w) and volume confirmation.
Enter long when price breaks above 6h Donchian upper band AND weekly pivot shows bullish bias (close > weekly PP) AND volume > 1.5x average.
Enter short when price breaks below 6h Donchian lower band AND weekly pivot shows bearish bias (close < weekly PP) AND volume > 1.5x average.
Exit when price returns to 6h Donchian midpoint (mean reversion) or opposite band break.
Designed for 6h timeframe with tight entries to avoid fee drag: target 12-30 trades/year.
Works in both bull and bear markets via weekly pivot filter (avoids counter-trend breakouts) and volume confirmation (reduces false breakouts).
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
    
    # Get 6h data for Donchian channels (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate Donchian(20) on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian upper band: highest high over 20 periods
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low over 20 periods
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint (mean)
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align 6h Donchian levels to 6h timeframe (identity alignment)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid)
    
    # Get 1w data for weekly pivot direction
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot point (PP) using previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift by 1 to get previous week's OHLC for current weekly pivot
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    # First value will be invalid (rolled from last), set to nan
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    # Weekly pivot point (PP)
    weekly_pp = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    
    # Align 1w weekly PP to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 6h Donchian (20), 1w PP (1), volume avg (20)
    start_idx = max(20, 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(weekly_pp_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_band = donchian_upper_aligned[i]
        lower_band = donchian_lower_aligned[i]
        mid_band = donchian_mid_aligned[i]
        weekly_pivot = weekly_pp_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Session filter: 00-24 UTC (trade all sessions for 6h)
        in_session = True  # 6h timeframe captures major sessions
        
        if position == 0:
            # Look for entry: Donchian breakout with weekly pivot direction and volume
            # Long: price breaks above upper band AND weekly close > weekly PP (bullish bias) AND volume AND session
            long_condition = (close_val > upper_band) and (close_val > weekly_pivot) and vol_conf and in_session
            # Short: price breaks below lower band AND weekly close < weekly PP (bearish bias) AND volume AND session
            short_condition = (close_val < lower_band) and (close_val < weekly_pivot) and vol_conf and in_session
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to midpoint OR breaks below lower band (failed breakout)
            exit_condition = (close_val <= mid_band) or (close_val < lower_band)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to midpoint OR breaks above upper band (failed breakout)
            exit_condition = (close_val >= mid_band) or (close_val > upper_band)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0