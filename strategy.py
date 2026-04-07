#!/usr/bin/env python3
"""
6h_weekly_pivot_breakout_1d_trend_volume_v1
Hypothesis: Weekly pivot levels provide strong support/resistance zones. Breakouts above
weekly R1 or below weekly S1 with volume confirmation and aligned with daily EMA200 trend
capture institutional flow. Weekly pivots adapt to volatility, working in both bull (buy
breakouts above R1 in uptrend) and bear (sell breakdowns below S1 in downtrend). Volume
filter ensures commitment, daily trend filter avoids counter-trend traps. Targets 15-25
trades/year by requiring confluence of weekly pivot breakout, volume surge, and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Daily EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Weekly data for pivot points (requires actual weekly bars)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot and support/resistance levels
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume confirmation: 24-period average (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if data not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Breakout conditions using weekly pivot levels
        breakout_up = close[i] > r1_aligned[i-1]
        breakout_down = close[i] < s1_aligned[i-1]
        
        # Daily trend filter
        above_daily_ema200 = close[i] > ema200_1d_aligned[i]
        below_daily_ema200 = close[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly S1 or below daily EMA200
            if close[i] < s1_aligned[i] or below_daily_ema200:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly R1 or above daily EMA200
            if close[i] > r1_aligned[i] or above_daily_ema200:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above weekly R1 with volume and above daily EMA200
            if breakout_up and vol_confirmed and above_daily_ema200:
                position = 1
                signals[i] = 0.25
            # Short: breakout below weekly S1 with volume and below daily EMA200
            elif breakout_down and vol_confirmed and below_daily_ema200:
                position = -1
                signals[i] = -0.25
    
    return signals