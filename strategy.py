#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout_Momentum
Hypothesis: Use weekly trend filter with daily Camarilla R1/S1 breakouts.
Long when daily price breaks above R1 with volume > 1.5x 20-bar avg AND weekly close > weekly open.
Short when daily price breaks below S1 with volume > 1.5x 20-bar avg AND weekly close < weekly open.
Exit when price crosses back through the pivot point (PP).
Designed for 1d timeframe to capture multi-day moves with ~10-25 trades/year.
Weekly trend filter reduces counter-trend trades in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Weekly trend: bullish if close > open
    weekly_bullish = df_weekly['close'] > df_weekly['open']
    weekly_bullish_vals = weekly_bullish.values
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bullish_vals)
    
    # Load daily data for Camarilla pivots
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Camarilla pivot levels (based on previous day)
    pp = np.full_like(close_daily, np.nan)
    r1 = np.full_like(close_daily, np.nan)
    s1 = np.full_like(close_daily, np.nan)
    
    for i in range(1, len(high_daily)):
        pp[i] = (high_daily[i-1] + low_daily[i-1] + close_daily[i-1]) / 3.0
        r1[i] = close_daily[i-1] + (high_daily[i-1] - low_daily[i-1]) * 1.1 / 12.0
        s1[i] = close_daily[i-1] - (high_daily[i-1] - low_daily[i-1]) * 1.1 / 12.0
    
    # Shift to align with current day (levels are based on previous day)
    pp = np.roll(pp, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    pp[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    
    pp_aligned = align_htf_to_ltf(prices, df_daily, pp)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above R1 + volume confirmation + weekly bullish
            if price > r1_aligned[i] and volume_ok and weekly_bullish_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 + volume confirmation + weekly bearish
            elif price < s1_aligned[i] and volume_ok and not weekly_bullish_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_Pivot_Breakout_Momentum"
timeframe = "1d"
leverage = 1.0