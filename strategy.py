#!/usr/bin/env python3
"""
6h_WeeklyPivot_DailyBreakout_v1
Hypothesis: On 6h timeframe, trade breakouts of weekly pivot levels (R1/S1) with daily trend alignment and volume confirmation. 
Weekly pivots provide strong structural support/resistance that works in both bull and bear markets. 
Daily EMA50 filter ensures we trade with the intermediate-term trend, reducing whipsaws. 
Volume spike confirms institutional participation. Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Get daily data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (R1, S1) from previous week
    prev_week_high = df_1w['high'].values
    prev_week_low = df_1w['low'].values
    prev_week_close = df_1w['close'].values
    
    # Typical price for weekly pivot
    weekly_pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_range = prev_week_high - prev_week_low
    
    # Weekly Camarilla R1 and S1 (main breakout levels)
    weekly_r1 = weekly_pp + weekly_range * 1.1 / 4.0
    weekly_s1 = weekly_pp - weekly_range * 1.1 / 4.0
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of weekly pivot (needs 1 week), daily EMA(50), volume MA (20)
    start_idx = max(1, 50, 20) + 1  # weekly data needs at least 1 bar
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        weekly_r1_val = weekly_r1_aligned[i]
        weekly_s1_val = weekly_s1_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price > daily EMA50 (uptrend) or < daily EMA50 (downtrend)
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        if position == 0:
            # Long: break above weekly R1 with uptrend and volume spike
            # Short: break below weekly S1 with downtrend and volume spike
            long_signal = (high_val > weekly_r1_val and uptrend and vol_spike)
            short_signal = (low_val < weekly_s1_val and downtrend and vol_spike)
            
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
            # Exit: trend reversal or price reaches weekly S1 (mean reversion target)
            if close_val < ema_50_1d_val or low_val <= weekly_s1_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend reversal or price reaches weekly R1 (mean reversion target)
            if close_val > ema_50_1d_val or high_val >= weekly_r1_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_DailyBreakout_v1"
timeframe = "6h"
leverage = 1.0