#!/usr/bin/env python3
"""
6h Weekly Pivot + Daily Trend + Volume Spike
Hypothesis: Price reacting to weekly pivot levels (R1/S1, R2/S2) with 1d EMA34 trend confirmation 
and volume spike produces high-probability mean-reversion or breakout trades. 
Weekly pivots act as strong support/resistance. In ranging markets, price reverts from R1/S1. 
In trending markets, breakouts through R2/S2 with volume continue the trend. 
Works in both bull/bear via trend filter. Targets 50-150 total trades over 4 years (12-37/year).
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
    
    # Get daily data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get weekly data for pivot points (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivots from prior week's OHLC
    # Using typical formula: P = (H+L+C)/3, R1 = 2*P-L, S1 = 2*P-H, etc.
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivots to 6h timeframe (no extra delay - pivots known at week close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate 20-period volume MA for 6h volume confirmation
    vol_ma_20_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_6h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and weekly data propagation
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        vol_ma_6h = vol_ma_20_6h[i]
        
        # Volume confirmation: current 6h volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma_6h
        
        # Distance to pivot levels as percentage of price
        dist_to_r1 = abs(curr_close - r1_val) / curr_close
        dist_to_s1 = abs(curr_close - s1_val) / curr_close
        dist_to_r2 = abs(curr_close - r2_val) / curr_close
        dist_to_s2 = abs(curr_close - s2_val) / curr_close
        
        # Threshold for considering price "at" a level (0.15%)
        near_threshold = 0.0015
        
        if position == 0:
            # Look for entry signals
            
            # Long scenarios:
            # 1. Mean reversion: price near S1/S2 in uptrend with volume spike
            long_mean_revert = ((curr_close <= s1_val * (1 + near_threshold) and 
                               curr_close >= s1_val * (1 - near_threshold)) or
                              (curr_close <= s2_val * (1 + near_threshold) and 
                               curr_close >= s2_val * (1 - near_threshold))) and \
                              ema_trend > pivot_val and volume_confirm
            
            # 2. Breakout continuation: price breaks above R2 with volume in uptrend
            long_breakout = (curr_close > r2_val and 
                           curr_close > r2_val * 1.001 and  # clear breakout
                           ema_trend > pivot_val and volume_confirm)
            
            # Short scenarios:
            # 1. Mean reversion: price near R1/R2 in downtrend with volume spike
            short_mean_revert = ((curr_close >= r1_val * (1 - near_threshold) and 
                                curr_close <= r1_val * (1 + near_threshold)) or
                               (curr_close >= r2_val * (1 - near_threshold) and 
                                curr_close <= r2_val * (1 + near_threshold))) and \
                                ema_trend < pivot_val and volume_confirm
            
            # 2. Breakdown continuation: price breaks below S2 with volume in downtrend
            short_breakout = (curr_close < s2_val and 
                            curr_close < s2_val * 0.999 and  # clear breakdown
                            ema_trend < pivot_val and volume_confirm)
            
            if long_mean_revert or long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_mean_revert or short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price reaches R1 (take profit) or breaks below S2 (stop) or trend changes
            if (curr_close >= r1_val or curr_close < s2_val or ema_trend < pivot_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price reaches S1 (take profit) or breaks above R2 (stop) or trend changes
            if (curr_close <= s1_val or curr_close > r2_val or ema_trend > pivot_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DailyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0