#!/usr/bin/env python3
"""
12h Monthly Pivot Breakout with Volume and Trend Filter
Strategy: Long when price breaks above monthly R1 with volume and above weekly EMA50;
          Short when price breaks below monthly S1 with volume and below weekly EMA50.
          Uses weekly EMA50 as trend filter to avoid counter-trend trades.
          Designed for low trade frequency with clear breakout edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate monthly pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get monthly data for pivot points (once before loop)
    df_monthly = get_htf_data(prices, '1M')
    
    # Calculate monthly pivot points
    monthly_high = df_monthly['high'].values
    monthly_low = df_monthly['low'].values
    monthly_close = df_monthly['close'].values
    
    _, monthly_r1, monthly_s1, _, _ = calculate_pivot_points(
        monthly_high, monthly_low, monthly_close
    )
    
    # Get weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_weekly['close'].values
    ema_50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align monthly and weekly data to 12h timeframe
    monthly_r1_aligned = align_htf_to_ltf(prices, df_monthly, monthly_r1)
    monthly_s1_aligned = align_htf_to_ltf(prices, df_monthly, monthly_s1)
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(monthly_r1_aligned[i]) or 
            np.isnan(monthly_s1_aligned[i]) or
            np.isnan(ema_50_weekly_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = monthly_r1_aligned[i]
        s1_level = monthly_s1_aligned[i]
        ema_50 = ema_50_weekly_aligned[i]
        
        if position == 0:
            # Long: break above monthly R1 with volume and above weekly EMA50
            if (price > r1_level and volume_spike[i] and price > ema_50):
                signals[i] = 0.25
                position = 1
            # Short: break below monthly S1 with volume and below weekly EMA50
            elif (price < s1_level and volume_spike[i] and price < ema_50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below monthly S1 or below weekly EMA50
            if price < s1_level or price < ema_50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above monthly R1 or above weekly EMA50
            if price > r1_level or price > ema_50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_MonthlyPivot_Breakout_Volume_WeeklyEMA50"
timeframe = "12h"
leverage = 1.0