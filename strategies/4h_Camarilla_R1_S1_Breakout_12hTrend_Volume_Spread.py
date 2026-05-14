#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_Spread
# Hypothesis: 4h chart strategy using Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume confirmation. 
# Uses spread between 12h EMA50 and price to avoid whipsaw in sideways markets. Designed for moderate trade frequency (20-50/year) to avoid fee drag.
# Combines proven elements: Camarilla levels, 12h trend filter, volume spike confirmation.

timeframe = "4h"
name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_Spread"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h closes for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate spread between price and 12h EMA50 (normalized by price)
    spread = (close - ema_50_12h_aligned) / close
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily high/low/close
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    camarilla_r1 = d_close + 1.1 * (d_high - d_low) / 12
    camarilla_s1 = d_close - 1.1 * (d_high - d_low) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike detection: 2x average volume (6-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6)  # Ensure we have EMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close > R1 with volume spike and price above 12h EMA50 (positive spread)
            if close[i] > camarilla_r1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and spread[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: close < S1 with volume spike and price below 12h EMA50 (negative spread)
            elif close[i] < camarilla_s1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and spread[i] < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: touch S1 (opposite level) or trend failure (price below 12h EMA50)
            if close[i] < camarilla_s1_aligned[i] or spread[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: touch R1 (opposite level) or trend failure (price above 12h EMA50)
            if close[i] > camarilla_r1_aligned[i] or spread[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals