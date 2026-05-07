#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Pivot_1wTrend_v1
Hypothesis: On 12h timeframe, use weekly Camarilla pivot levels (R3/S3) for breakout entries, filtered by daily EMA trend and volume spikes. Long when price breaks above R3 with daily uptrend and volume spike. Short when price breaks below S3 with daily downtrend and volume spike. Weekly pivots provide stronger support/resistance than daily, reducing false breaks. EMA filter ensures alignment with higher timeframe trend. Volume spike confirms breakout strength. Designed for fewer trades (target 12-37/year) to minimize fee drag on 12h timeframe.
"""
name = "12h_Camarilla_R3S3_Pivot_1wTrend_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on prior week)
    # Using typical pivot formula: P = (H+L+C)/3
    # R3 = P + 1.1*(H-L), S3 = P - 1.1*(H-L)
    ph = df_1w['high'].values
    pl = df_1w['low'].values
    pc = df_1w['close'].values
    
    p = (ph + pl + pc) / 3.0
    r3 = p + 1.1 * (ph - pl)
    s3 = p - 1.1 * (ph - pl)
    
    # Align weekly levels to 12h timeframe (wait for weekly close)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 5 bars between trades (60 hours on 12h TF) to reduce frequency
            if bars_since_exit < 5:
                continue
                
            # Long: price breaks above R3 + daily uptrend + volume filter
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S3 + daily downtrend + volume filter
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to weekly pivot point (P)
            # Calculate weekly pivot point for exit
            pp = (ph + pl + pc) / 3.0
            pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
            
            if not np.isnan(pp_aligned[i]):
                if position == 1 and close[i] < pp_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                elif position == -1 and close[i] > pp_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    # Hold position
                    signals[i] = 0.25 if position == 1 else -0.25
            else:
                # Hold if pivot not ready
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals