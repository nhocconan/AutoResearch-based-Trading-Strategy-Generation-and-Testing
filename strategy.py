#!/usr/bin/env python3
"""
12h Camarilla R4/S4 Breakout with 1d Trend Filter and Volume Spike
- Uses Camarilla levels from daily timeframe (S4/S3 for long, R4/R3 for short)
- Breakout above S4 with 1d uptrend or below R4 with 1d downtrend
- Volume spike confirms breakout strength
- Works in bull/bear by using 1d trend filter to avoid counter-trend trades
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R4S4_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    # S3 = C - (H-L)*1.16, S4 = C - (H-L)*1.33, R3 = C + (H-L)*1.16, R4 = C + (H-L)*1.33
    n1d = len(close_1d)
    camarilla_S3 = np.full(n1d, np.nan)
    camarilla_S4 = np.full(n1d, np.nan)
    camarilla_R3 = np.full(n1d, np.nan)
    camarilla_R4 = np.full(n1d, np.nan)
    
    for i in range(1, n1d):
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        range_val = H - L
        camarilla_S3[i] = C - range_val * 1.16
        camarilla_S4[i] = C - range_val * 1.33
        camarilla_R3[i] = C + range_val * 1.16
        camarilla_R4[i] = C + range_val * 1.33
    
    # Align Camarilla levels to 12h timeframe
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    # 1d data for trend filter (EMA50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_S4_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_R4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above S4 (strong support) with 1d uptrend + volume spike
            long_cond = (close[i] > camarilla_S4_aligned[i] and 
                        ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below R4 (strong resistance) with 1d downtrend + volume spike
            short_cond = (close[i] < camarilla_R4_aligned[i] and 
                         ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (weaker support)
            if close[i] < camarilla_S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 (weaker resistance)
            if close[i] > camarilla_R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals