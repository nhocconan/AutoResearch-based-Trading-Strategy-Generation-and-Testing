#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with volume confirmation and weekly trend filter
    # Long: price breaks above H3 (resistance) AND volume > 1.5x 20-period average AND weekly close > weekly open
    # Short: price breaks below L3 (support) AND volume > 1.5x 20-period average AND weekly close < weekly open
    # Exit: price returns to PIVOT point
    # Using 1d for signal generation, 1w for trend filter to reduce counter-trend trades
    # Discrete position sizing (0.25) to balance return and risk
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data for Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivots (based on previous 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # PIVOT = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # RANGE = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels:
    # H3 = C + RANGE * 1.1/4
    # L3 = C - RANGE * 1.1/4
    h3_1d = close_1d + range_1d * 1.1 / 4
    l3_1d = close_1d - range_1d * 1.1 / 4
    
    # Align 1d Camarilla levels to lower timeframe (wait for completed 1d bar)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Get 1w data for trend filter (weekly bull/bear)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        # Fallback to neutral if 1w not enough data
        weekly_bull = np.full(n, True)  # neutral - no filter
        weekly_bear = np.full(n, True)  # neutral - no filter
    else:
        open_1w = df_1w['open'].values
        close_1w = df_1w['close'].values
        # Weekly bull = close > open, Weekly bear = close < open
        weekly_bull_raw = close_1w > open_1w
        weekly_bear_raw = close_1w < open_1w
        # Align to lower timeframe (wait for completed 1w bar)
        weekly_bull = align_htf_to_ltf(prices, df_1w, weekly_bull_raw.astype(float)) > 0.5
        weekly_bear = align_htf_to_ltf(prices, df_1w, weekly_bear_raw.astype(float)) > 0.5
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Weekly trend filter
        long_trend_ok = weekly_bull[i] if i < len(weekly_bull) else True
        short_trend_ok = weekly_bear[i] if i < len(weekly_bear) else True
        
        # Entry logic: Camarilla breakout + volume + weekly trend
        long_entry = (close[i] > h3_1d_aligned[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < l3_1d_aligned[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: return to pivot
        long_exit = close[i] < pivot_1d_aligned[i]
        short_exit = close[i] > pivot_1d_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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

name = "1d_1w_camarilla_breakout_volume_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0