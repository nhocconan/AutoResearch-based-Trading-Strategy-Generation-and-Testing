#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_R1S1_Breakout_Volume_Trend
Strategy: Daily Camarilla pivot R1/S1 breakout with volume confirmation and weekly trend filter.
Long: Price breaks above daily R1 + volume > 1.5x 20-day avg + price above weekly EMA34
Short: Price breaks below daily S1 + volume > 1.5x 20-day avg + price below weekly EMA34
Exit: Price returns to daily pivot point (PP)
Position size: 0.25
Designed to capture institutional breakouts aligned with weekly trend in both bull and bear markets.
Timeframe: 1d
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily pivot points (PP, R1, S1)
    # Using daily OHLC from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_series_1w = pd.Series(close_1w)
    ema34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all to daily timeframe
    pivot_pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1d, ema34_1w)
    
    # Volume confirmation: 20-day average volume
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need weekly EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA34
        price_above_ema = close[i] > ema34_1w_aligned[i]
        price_below_ema = close[i] < ema34_1w_aligned[i]
        
        # Breakout conditions (using previous day's levels to avoid look-ahead)
        breakout_up = close[i] > r1_aligned[i-1]  # break above R1
        breakout_down = close[i] < s1_aligned[i-1]  # break below S1
        
        # Exit condition: return to pivot point (PP)
        return_to_pp = abs(close[i] - pivot_pp_aligned[i]) < 0.003 * close[i]  # within 0.3% of PP
        
        if position == 0:
            # Long: breakout above R1 + volume filter + price above weekly EMA34
            if breakout_up and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + price below weekly EMA34
            elif breakout_down and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to PP or break below S1
            if return_to_pp or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to PP or break above R1
            if return_to_pp or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_Pivot_R1S1_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0