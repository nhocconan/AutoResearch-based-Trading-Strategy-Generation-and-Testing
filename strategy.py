#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_VolumeATRFilter_V1
# Hypothesis: Camarilla pivot levels (R1/S1) on daily timeframe act as key support/resistance. Breakouts above R1 or below S1 with volume confirmation and ATR-based volatility filter capture momentum. The 12h timeframe reduces trade frequency to avoid fee drag, while weekly trend filter ensures alignment with higher-timeframe momentum. Designed to work in both bull and bear markets by focusing on institutional pivot levels.

name = "12h_Camarilla_R1_S1_Breakout_VolumeATRFilter_V1"
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
    
    # Get daily data for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # R1 = PP + (Range * 1.1 / 12)
    # S1 = PP - (Range * 1.1 / 12)
    r1 = pp + (range_1d * 1.1 / 12)
    s1 = pp - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate ATR (14-period) for volatility filter
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.full_like(high_1d, np.nan)
    for i in range(len(tr_1d)):
        if i >= 13:  # 14-period ATR
            atr_1d[i] = np.nanmean(tr_1d[i-13:i+1])
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Weekly trend filter: EMA(34) on weekly closes
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average on 12h chart
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure weekly EMA and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + above weekly EMA34 + volume confirmation + volatility filter
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_filter[i] and
                atr_1d_aligned[i] > 0):  # volatility present
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + below weekly EMA34 + volume confirmation + volatility filter
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_filter[i] and
                  atr_1d_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or weekly EMA turns down
            if close[i] < s1_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or weekly EMA turns up
            if close[i] > r1_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals