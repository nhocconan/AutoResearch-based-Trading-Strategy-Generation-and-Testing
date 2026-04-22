#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla pivot reversal with 1-day volume spike and 1-week EMA trend filter.
Long when price touches S1 level with volume > 1.5x 20-day average and weekly EMA50 rising.
Short when price touches R1 level with volume > 1.5x 20-day average and weekly EMA50 falling.
Exit when price reaches opposite Camarilla level or closes beyond pivot point.
Camarilla levels provide high-probability reversal zones; volume spike confirms institutional interest;
weekly EMA filters for higher timeframe trend alignment. Designed for low trade frequency.
Works in both bull and bear markets by fading extremes in ranging conditions while respecting weekly trend.
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
    
    # Load 1-day data for pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1-week data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # R2 = Close + 0.5 * (High - Low)
    # R1 = Close + 0.25 * (High - Low)
    # PP = (High + Low + Close) / 3
    # S1 = Close - 0.25 * (High - Low)
    # S2 = Close - 0.5 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    
    # We only need R1 and S1 for reversal strategy
    camarilla_r1 = close_1d + 0.25 * (high_1d - low_1d)
    camarilla_s1 = close_1d - 0.25 * (high_1d - low_1d)
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 20-period average volume for volume spike filter
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for volume MA
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches or goes below S1 with volume spike and weekly EMA rising
            if (low[i] <= camarilla_s1_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i] and
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price touches or goes above R1 with volume spike and weekly EMA falling
            elif (high[i] >= camarilla_r1_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i] and
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price reaches or goes above pivot point OR reaches R1
                if (high[i] >= camarilla_pp_aligned[i] or 
                    high[i] >= camarilla_r1_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price reaches or goes below pivot point OR reaches S1
                if (low[i] <= camarilla_pp_aligned[i] or 
                    low[i] <= camarilla_s1_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_Reversal_VolumeSpike_1wEMA50_Trend"
timeframe = "4h"
leverage = 1.0