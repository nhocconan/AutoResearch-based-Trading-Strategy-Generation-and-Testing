#!/usr/bin/env python3
"""
6h Weekly Pivot + Daily Trend + Volume Confirmation
Hypothesis: Weekly pivot points (PP, R1, S1) act as strong support/resistance on 6h chart.
Trades are taken in direction of 1d EMA34 trend: long when price > EMA34 and breaks above R1,
short when price < EMA34 and breaks below S1. Volume spike confirms institutional participation.
Weekly pivots provide structure that works in both bull/bear markets, while EMA34 filter
ensures we trade with higher timeframe momentum. Designed for low trade frequency
(12-37/year) to minimize fee drag on 6h timeframe.
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
    
    # Get weekly data for pivot points (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly OHLC
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Weekly pivot point: PP = (H + L + C) / 3
    weekly_pp = (w_high + w_low + w_close) / 3.0
    # Weekly R1 = (2 * PP) - L
    weekly_r1 = (2 * weekly_pp) - w_low
    # Weekly S1 = (2 * PP) - H
    weekly_s1 = (2 * weekly_pp) - w_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA
    start_idx = max(34, 20)
    
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
        r1_level = weekly_r1_aligned[i]
        s1_level = weekly_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R1 AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (curr_close > r1_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below S1 AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (curr_close < s1_level) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below weekly PP (reversal) OR price crosses below EMA (trend change)
            if (curr_close < weekly_pp_aligned[i]) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above weekly PP (reversal) OR price crosses above EMA (trend change)
            if (curr_close > weekly_pp_aligned[i]) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0