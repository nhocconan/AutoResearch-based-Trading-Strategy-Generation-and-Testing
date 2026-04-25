#!/usr/bin/env python3
"""
6h Weekly Pivot + Daily Volume Spike + 1d EMA50 Trend Filter
Hypothesis: Weekly pivot levels (from prior week) act as strong support/resistance. 
Breakouts above R1 or below S1 with daily volume spike (>2.0x 20-bar vol MA) and 
1d EMA50 trend alignment capture strong moves. In ranging markets (price between S1/R1), 
we fade extremes. Designed for BTC/ETH on 6h timeframe with 50-150 trades over 4 years 
(12-37/year) to minimize fee drag while maintaining edge in both bull and bear regimes.
Weekly pivots provide structure that works across market cycles.
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
    
    # Get 1d data for EMA50 trend filter and volume MA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 50 for EMA + 2 for safety
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (using 1d volume)
    vol_1d = pd.Series(df_1d['volume'])
    vol_ma_20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get 1w data for weekly pivot points (prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior week's high, low, close for weekly pivot calculation
    prev_week_high = df_1w['high'].shift(1).values  # Shift to get prior week
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Align to 6h timeframe
    prev_week_high_6h = align_htf_to_ltf(prices, df_1w, prev_week_high)
    prev_week_low_6h = align_htf_to_ltf(prices, df_1w, prev_week_low)
    prev_week_close_6h = align_htf_to_ltf(prices, df_1w, prev_week_close)
    
    # Calculate weekly pivot points and support/resistance levels
    # Pivot = (High + Low + Close) / 3
    # R1 = (2 * Pivot) - Low
    # S1 = (2 * Pivot) - High
    # R2 = Pivot + (High - Low)
    # S2 = Pivot - (High - Low)
    # R3 = High + 2*(Pivot - Low)
    # S3 = Low - 2*(High - Pivot)
    pp = (prev_week_high_6h + prev_week_low_6h + prev_week_close_6h) / 3.0
    r1 = (2 * pp) - prev_week_low_6h
    s1 = (2 * pp) - prev_week_high_6h
    r2 = pp + (prev_week_high_6h - prev_week_low_6h)
    s2 = pp - (prev_week_high_6h - prev_week_low_6h)
    r3 = prev_week_high_6h + 2 * (pp - prev_week_low_6h)
    s3 = prev_week_low_6h - 2 * (prev_week_high_6h - pp)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50, volume MA, and weekly pivots
    start_idx = max(52, 20)  # 52 for EMA50 (50 + 2 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(r1[i]) or 
            np.isnan(s1[i]) or
            np.isnan(r2[i]) or 
            np.isnan(s2[i]) or
            np.isnan(r3[i]) or 
            np.isnan(s3[i]) or
            np.isnan(pp[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        r1_val = r1[i]
        s1_val = s1[i]
        r2_val = r2[i]
        s2_val = s2[i]
        r3_val = r3[i]
        s3_val = s3[i]
        pp_val = pp[i]
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            if price_above_ema:
                # Uptrend: look for long breakouts above R1/R2/R3
                long_signal = ((curr_close > r1_val) or (curr_close > r2_val) or (curr_close > r3_val)) and volume_confirm
            else:
                # Downtrend: look for short breakdowns below S1/S2/S3
                short_signal = ((curr_close < s1_val) or (curr_close < s2_val) or (curr_close < s3_val)) and volume_confirm
            
            # In ranging markets (price between S1/R1), fade extremes
            in_range = (curr_close >= s1_val) and (curr_close <= r1_val)
            if in_range:
                # Fade extremes: long near S1, short near R1
                long_signal = (curr_close <= s1_val * 1.002) and volume_confirm  # near S1
                short_signal = (curr_close >= r1_val * 0.998) and volume_confirm  # near R1
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            # Clear signal flags for next iteration
            if 'long_signal' in locals():
                del long_signal
            if 'short_signal' in locals():
                del short_signal
        elif position == 1:
            # Exit long: price breaks below S1 or reverses below EMA
            if curr_close < s1_val or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 or reverses above EMA
            if curr_close > r1_val or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1S1_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0