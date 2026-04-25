#!/usr/bin/env python3
"""
6h Weekly Pivot R1S1 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Weekly R1/S1 pivot levels act as significant support/resistance. 
Breakouts above weekly R1 or below weekly S1 with volume confirmation and aligned 
1d EMA34 trend capture institutional moves in both bull and bear markets. The 1d 
EMA34 ensures we trade with higher timeframe momentum, reducing false breakouts. 
Volume spike confirms participation. Designed for moderate trade frequency (12-37/year) 
on 6h timeframe.
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for Weekly pivot levels (R1, S1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Weekly pivot levels from previous week's OHLC
    # Use shift(1) to ensure we only use completed weekly bars (no look-ahead)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Weekly pivot point and R1/S1 levels
    weekly_pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_r1 = 2 * weekly_pp - prev_week_low
    weekly_s1 = 2 * weekly_pp - prev_week_high
    
    # Align Weekly pivot levels to 6h timeframe (completed 1w bar only)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA, and prior week data
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        weekly_r1 = weekly_r1_aligned[i]
        weekly_s1 = weekly_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Weekly R1 (strong resistance) AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (curr_close > weekly_r1) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Weekly S1 (strong support) AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (curr_close < weekly_s1) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below Weekly S1 (support) OR price crosses below EMA (trend change)
            if (curr_close < weekly_s1) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Weekly R1 (resistance) OR price crosses above EMA (trend change)
            if (curr_close > weekly_r1) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0