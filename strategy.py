#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: Camarilla pivot levels from 1d with volume confirmation and 1d EMA trend filter.
Long when price touches or breaks above Camarilla R3 with volume > average and price above 1d EMA50.
Short when price touches or breaks below Camarilla S3 with volume > average and price below 1d EMA50.
Designed for 12-37 trades/year on 12h with clear logic that works in bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla formulas: 
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    # Pivot point
    PP = (H + L + C) / 3.0
    # Range
    range_hl = H - L
    
    # Camarilla levels
    R3 = C + (range_hl * 1.1 / 4.0)
    S3 = C - (range_hl * 1.1 / 4.0)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(C).ewm(span=50, adjust=False).mean().values
    
    # Align 1d data to 12h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Price touching/breaking Camarilla levels
        touch_R3 = close[i] >= R3_aligned[i]
        touch_S3 = close[i] <= S3_aligned[i]
        
        # 1d trend filter
        above_1d_ema50 = close[i] > ema50_1d_aligned[i]
        below_1d_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 or trend turns bearish
            if close[i] < S3_aligned[i] or below_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 or trend turns bullish
            if close[i] > R3_aligned[i] or above_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches/breaks above R3 with volume confirmation and bullish trend
            if touch_R3 and vol_confirmed and above_1d_ema50:
                position = 1
                signals[i] = 0.25
            # Short: price touches/breaks below S3 with volume confirmation and bearish trend
            elif touch_S3 and vol_confirmed and below_1d_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals