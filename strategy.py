#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Reversal_Strategy_v1
Hypothesis: Weekly Camarilla pivot levels (S1/S2/R1/R2) act as strong support/resistance.
Buy near S1/S2 in uptrend, sell near R1/R2 in downtrend, using daily EMA filter and volume confirmation.
Designed for low trade frequency (<20/year) on 1d timeframe to minimize fee drag and work in both bull/bear markets.
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
    
    # Get weekly data for Camarilla pivots (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r4 = close_1w + ((high_1w - low_1w) * 1.5000)
    r3 = close_1w + ((high_1w - low_1w) * 1.2500)
    r2 = close_1w + ((high_1w - low_1w) * 1.1666)
    r1 = close_1w + ((high_1w - low_1w) * 1.0833)
    s1 = close_1w - ((high_1w - low_1w) * 1.0833)
    s2 = close_1w - ((high_1w - low_1w) * 1.1666)
    s3 = close_1w - ((high_1w - low_1w) * 1.2500)
    s4 = close_1w - ((high_1w - low_1w) * 1.5000)
    
    # Align to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: near S1/S2 with uptrend and volume
            if ((abs(price - s1_aligned[i]) < 0.005 * s1_aligned[i] or
                 abs(price - s2_aligned[i]) < 0.005 * s2_aligned[i]) and
                price > ema_trend and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: near R1/R2 with downtrend and volume
            elif ((abs(price - r1_aligned[i]) < 0.005 * r1_aligned[i] or
                   abs(price - r2_aligned[i]) < 0.005 * r2_aligned[i]) and
                  price < ema_trend and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price reaches R1/R2 or breaks below EMA
            if (price >= r1_aligned[i] or price >= r2_aligned[i] or
                price < ema_trend):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price reaches S1/S2 or breaks above EMA
            if (price <= s1_aligned[i] or price <= s2_aligned[i] or
                price > ema_trend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_Reversal_Strategy_v1"
timeframe = "1d"
leverage = 1.0