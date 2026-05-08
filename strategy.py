#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 breakout with weekly trend filter and volume spike
- Uses weekly trend (EMA50) to avoid counter-trend trades in any market regime
- Enters on breakout of R4 (short) or S4 (long) with volume confirmation
- Weekly trend filter ensures alignment with higher timeframe momentum
- Designed for low trade frequency (<50/year) to minimize fee drag on 4h timeframe
- Works in bull/bear by using weekly trend to filter direction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R4S4_Breakout_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's data
    # S4 = C - (H-L)*1.50, R4 = C + (H-L)*1.50
    n1d = len(close_1d)
    camarilla_S4 = np.full(n1d, np.nan)
    camarilla_R4 = np.full(n1d, np.nan)
    
    for i in range(1, n1d):
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        range_val = H - L
        camarilla_S4[i] = C - range_val * 1.50
        camarilla_R4[i] = C + range_val * 1.50
    
    # Align Camarilla levels to 4h timeframe
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_S4_aligned[i]) or np.isnan(camarilla_R4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above S4 (strong support) with weekly uptrend + volume spike
            long_cond = (close[i] > camarilla_S4_aligned[i] and 
                        ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below R4 (strong resistance) with weekly downtrend + volume spike
            short_cond = (close[i] < camarilla_R4_aligned[i] and 
                         ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (weaker support) or weekly trend turns down
            if close[i] < camarilla_S4_aligned[i] * 0.995 or ema_50_1w_aligned[i] < ema_50_1w_aligned[i-5]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R4 (strong resistance) or weekly trend turns up
            if close[i] > camarilla_R4_aligned[i] * 1.005 or ema_50_1w_aligned[i] > ema_50_1w_aligned[i-5]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals