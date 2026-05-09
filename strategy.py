#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R4S4_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for R4/S4 pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot and levels from previous week's OHLC
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    prev_weekly_range = prev_high_1w - prev_low_1w
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r4_1w = pivot_1w + 1.1 * prev_weekly_range * 1.1  # R4 = pivot + 1.1*range*1.1
    s4_1w = pivot_1w - 1.1 * prev_weekly_range * 1.1  # S4 = pivot - 1.1*range*1.1
    
    # Align weekly R4/S4 to 6h
    r4_6h = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_6h = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection (24-period for 6h: approx 4 days)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(ema34_6h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 24-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above weekly R4 with uptrend and volume spike
            if close[i] > r4_6h[i] and close[i] > ema34_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S4 with downtrend and volume spike
            elif close[i] < s4_6h[i] and close[i] < ema34_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below weekly S4 OR trend turns down
            if close[i] < s4_6h[i] or close[i] < ema34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above weekly R4 OR trend turns up
            if close[i] > r4_6h[i] or close[i] > ema34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals