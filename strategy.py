#!/usr/bin/env python3
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
    
    # Get 1d data for weekly pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot levels from prior week (using 1d data)
    # Pivot point = (H + L + C)/3, Support/Resistance levels
    pp = np.full_like(close_1d, np.nan)
    r1 = np.full_like(close_1d, np.nan)
    s1 = np.full_like(close_1d, np.nan)
    r2 = np.full_like(close_1d, np.nan)
    s2 = np.full_like(close_1d, np.nan)
    r3 = np.full_like(close_1d, np.nan)
    s3 = np.full_like(close_1d, np.nan)
    r4 = np.full_like(close_1d, np.nan)
    s4 = np.full_like(close_1d, np.nan)
    
    for i in range(6, len(close_1d)):  # Need 5 prior days for weekly calculation
        # Use prior 5 trading days (week) - excluding current day
        week_high = np.max(high_1d[i-5:i])
        week_low = np.min(low_1d[i-5:i])
        week_close = close_1d[i-1]  # Previous day's close
        
        pp[i] = (week_high + week_low + week_close) / 3.0
        r1[i] = 2 * pp[i] - week_low
        s1[i] = 2 * pp[i] - week_high
        r2[i] = pp[i] + (week_high - week_low)
        s2[i] = pp[i] - (week_high - week_low)
        r3[i] = week_high + 2 * (pp[i] - week_low)
        s3[i] = week_low - 2 * (week_high - pp[i])
        r4[i] = pp[i] + 3 * (week_high - week_low)
        s4[i] = pp[i] - 3 * (week_high - week_low)
    
    # Calculate 50-period EMA on 1d for trend filter
    if len(close_1d) >= 50:
        ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_1d = np.full_like(close_1d, np.nan)
    
    # Align all data to 6h timeframe
    pp_6h = align_htf_to_ltf(prices, df_1d, pp)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 6 + 1  # Need prior week data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 50-day EMA
        uptrend = close[i] > ema_1d_6h[i]
        downtrend = close[i] < ema_1d_6h[i]
        
        if position == 0:
            # Long: price breaks above R4 with uptrend (strong bullish breakout)
            if close[i] > r4_6h[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with downtrend (strong bearish breakdown)
            elif close[i] < s4_6h[i] and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below R1 (profit target) OR trend reverses
            if close[i] < r1_6h[i] or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above S1 (profit target) OR trend reverses
            if close[i] > s1_6h[i] or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R4S4_Breakout_50EMA"
timeframe = "6h"
leverage = 1.0