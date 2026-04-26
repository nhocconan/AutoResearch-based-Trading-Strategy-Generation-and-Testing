#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike
Hypothesis: Use 6h Donchian(20) breakouts aligned with weekly pivot direction (from prior weekly Camarilla R4/S4 levels) and volume confirmation (>2.0x 20-period average). Weekly pivot provides structural bias from longer timeframe, Donchian captures breakouts, volume confirms conviction. Designed for 6h timeframe to target 12-37 trades/year (50-150 over 4 years). Discrete sizing 0.25. Works in both bull and bear markets by aligning with weekly structure.
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
    open_time = prices['open_time'].values
    
    # Session filter: UTC 8-20 for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d and 1w data for Camarilla calculation (weekly pivot from prior weekly)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly Camarilla R4 and S4 from prior weekly bar (for directional bias)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    if len(high_1w) < 2:
        camarilla_r4_1w = np.full_like(close_1w, np.nan)
        camarilla_s4_1w = np.full_like(close_1w, np.nan)
    else:
        # Camarilla R4 = close + 1.1*(high-low)*1.1/2, S4 = close - 1.1*(high-low)*1.1/2
        camarilla_r4_1w = close_1w[:-1] + 1.1 * (high_1w[:-1] - low_1w[:-1]) * 1.1 / 2
        camarilla_s4_1w = close_1w[:-1] - 1.1 * (high_1w[:-1] - low_1w[:-1]) * 1.1 / 2
        camarilla_r4_1w = np.concatenate([[np.nan], camarilla_r4_1w])
        camarilla_s4_1w = np.concatenate([[np.nan], camarilla_s4_1w])
    
    # Align weekly Camarilla levels to 6h
    camarilla_r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    
    # 6h Donchian(20) channels
    def donchian_channels(high_arr, low_arr, window):
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian (20), weekly Camarilla (need 2 bars), volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(camarilla_r4_1w_aligned[i]) or 
            np.isnan(camarilla_s4_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        dc_upper_val = donchian_upper[i]
        dc_lower_val = donchian_lower[i]
        r4_1w_val = camarilla_r4_1w_aligned[i]
        s4_1w_val = camarilla_s4_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian upper AND weekly bias is bullish (close > weekly R4)
            long_breakout = close_val > dc_upper_val
            weekly_bullish = close_val > r4_1w_val
            long_signal = long_breakout and weekly_bullish and volume_confirmed
            
            # Short: price breaks below Donchian lower AND weekly bias is bearish (close < weekly S4)
            short_breakout = close_val < dc_lower_val
            weekly_bearish = close_val < s4_1w_val
            short_signal = short_breakout and weekly_bearish and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below Donchian lower (breakdown)
            if close_val < dc_lower_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: weekly bias turns bearish (close below weekly S4)
            elif close_val < s4_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above Donchian upper (breakout)
            if close_val > dc_upper_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: weekly bias turns bullish (close above weekly R4)
            elif close_val > r4_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike"
timeframe = "6h"
leverage = 1.0