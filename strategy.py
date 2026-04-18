#!/usr/bin/env python3
"""
4h Pivot Range Breakout with Volume Spike and 1-day Trend Filter
Breakout above/below daily pivot levels (R1/S1) + volume spike + daily EMA trend filter
Designed to capture momentum with low trade frequency and work in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_ktf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Calculate daily EMA20 for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align pivot levels and EMA to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume spike detection (2x 24-period average - 4 days worth)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_20_1d_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume spike + above daily EMA
            if (price > r1_val and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume spike + below daily EMA
            elif (price < s1_val and 
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 or trend reversal
            if price < s1_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above R1 or trend reversal
            if price > r1_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Breakout_Volume_DailyTrend"
timeframe = "4h"
leverage = 1.0