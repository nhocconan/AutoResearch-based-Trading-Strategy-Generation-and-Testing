#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R4_S4_Breakout_1dTrend_Adaptive"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get weekly data (actual weekly candles)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pp_w = (high_w + low_w + close_w) / 3
    r4_w = pp_w + (high_w - low_w) * 1.5  # R4 = PP + 1.5*(High-Low)
    s4_w = pp_w - (high_w - low_w) * 1.5  # S4 = PP - 1.5*(High-Low)
    
    # Align weekly pivots to 6h timeframe
    pp_w_aligned = align_htf_to_ltf(prices, df_1w, pp_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_1w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_1w, s4_w)
    
    # Use previous period's values to avoid look-ahead
    r4_w_prev = np.roll(r4_w_aligned, 1)
    s4_w_prev = np.roll(s4_w_aligned, 1)
    r4_w_prev[0] = np.nan
    s4_w_prev[0] = np.nan
    
    # Volume confirmation: current volume > 2.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r4_w_prev[i]) or 
            np.isnan(s4_w_prev[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Close breaks above R4 with volume spike and above 1d EMA trend
            if close[i] > r4_w_prev[i] and vol_ok and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S4 with volume spike and below 1d EMA trend
            elif close[i] < s4_w_prev[i] and vol_ok and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below weekly pivot point (mean reversion)
            if close[i] < pp_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back above weekly pivot point
            if close[i] > pp_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals