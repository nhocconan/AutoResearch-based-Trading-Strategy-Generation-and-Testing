# -*- coding: utf-8 -*-
# 6h_1w_1d_camarilla_breakout_weekly_trend_v1
# Hypothesis: 6-hour strategy using weekly trend from Ichimoku (Tenkan-Kijun cross) on 1w,
# combined with daily Camarilla breakouts (H3/L3) and volume confirmation on 6h.
# Weekly trend ensures we only trade in the direction of the higher timeframe trend,
# reducing false breakouts in sideways markets. Volume filter ensures breakout strength.
# Designed for low trade frequency (10-30/year) to minimize fee drag in 6h timeframe.
# Works in both bull and bear markets by following weekly trend direction.

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
    
    # Get weekly data for Ichimoku trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Ichimoku on weekly data (using proper formulas)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = np.array([np.max(high_1w[i-8:i+1]) if i >= 8 else np.nan for i in range(len(high_1w))])
    period9_low = np.array([np.min(low_1w[i-8:i+1]) if i >= 8 else np.nan for i in range(len(low_1w))])
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = np.array([np.max(high_1w[i-25:i+1]) if i >= 25 else np.nan for i in range(len(high_1w))])
    period26_low = np.array([np.min(low_1w[i-25:i+1]) if i >= 25 else np.nan for i in range(len(low_1w))])
    kijun = (period26_high + period26_low) / 2
    
    # Weekly trend: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
    weekly_trend = np.where(tenkan > kijun, 1, np.where(tenkan < kijun, -1, 0))
    
    # Align weekly trend to 6h timeframe
    weekly_trend_6h = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Calculate daily Camarilla levels using PREVIOUS day's data (no look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first day's previous values to NaN (no data yet)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations using previous day's data
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + 1.1 * range_val / 2
    l3 = pivot - 1.1 * range_val / 2
    h4 = pivot + 1.1 * range_val
    l4 = pivot - 1.1 * range_val
    
    # Align Camarilla levels to 6h timeframe
    h3_6h = align_htf_to_ltf(prices, df_1d, h3)
    l3_6h = align_htf_to_ltf(prices, df_1d, l3)
    h4_6h = align_htf_to_ltf(prices, df_1d, h4)
    l4_6h = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(weekly_trend_6h[i]) or np.isnan(h3_6h[i]) or np.isnan(l3_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average (moderate filter)
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Only take trades in direction of weekly trend
        weekly_bullish = weekly_trend_6h[i] == 1
        weekly_bearish = weekly_trend_6h[i] == -1
        
        # Entry conditions: Camarilla H3/L3 breakout with volume confirmation and weekly trend alignment
        long_breakout = (close[i] > h3_6h[i]) and volume_filter and weekly_bullish
        short_breakout = (close[i] < l3_6h[i]) and volume_filter and weekly_bearish
        
        # Exit conditions: touch opposite H3/L3 level or weekly trend reversal
        long_exit = (close[i] < l3_6h[i]) or (weekly_trend_6h[i] == -1)
        short_exit = (close[i] > h3_6h[i]) or (weekly_trend_6h[i] == 1)
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "6h_1w_1d_camarilla_breakout_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0