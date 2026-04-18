#!/usr/bin/env python3
"""
4h 12h/1d Pivot Breakout with Volume Confirmation
Hypothesis: Price breaking above/below 1d pivot levels with volume confirmation (volume > 1.3x average) 
and 12h EMA trend alignment provides strong momentum signals. Works in both bull and bear markets 
by capturing breakouts from key support/resistance levels.
Target: 15-30 trades/year to minimize fee drain.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter (updated every 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Get 1d data for pivot levels (updated daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate daily pivot levels (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume confirmation: volume > 1.3x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema34_12h_aligned[i]
        vol_conf = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: price breaks above R1 with volume and 12h uptrend
            if price > r1_aligned[i] and vol_conf and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and 12h downtrend
            elif price < s1_aligned[i] and vol_conf and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to pivot or trend breaks
            if price < pivot_aligned[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to pivot or trend breaks
            if price > pivot_aligned[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Volume_12hEMA34"
timeframe = "4h"
leverage = 1.0