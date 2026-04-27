#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R4 = Close + ((High - Low) * 1.5)
    # R3 = Close + ((High - Low) * 1.125)
    # R2 = Close + ((High - Low) * 0.75)
    # R1 = Close + ((High - Low) * 0.5)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 0.5)
    # S2 = Close - ((High - Low) * 0.75)
    # S3 = Close - ((High - Low) * 1.125)
    # S4 = Close - ((High - Low) * 1.5)
    
    r4_1d = close_1d + (high_1d - low_1d) * 1.5
    r3_1d = close_1d + (high_1d - low_1d) * 1.125
    r2_1d = close_1d + (high_1d - low_1d) * 0.75
    r1_1d = close_1d + (high_1d - low_1d) * 0.5
    pp_1d = (high_1d + low_1d + close_1d) / 3
    s1_1d = close_1d - (high_1d - low_1d) * 0.5
    s2_1d = close_1d - (high_1d - low_1d) * 0.75
    s3_1d = close_1d - (high_1d - low_1d) * 1.125
    s4_1d = close_1d - (high_1d - low_1d) * 1.5
    
    # Align Camarilla levels to 12h timeframe
    r4_12h = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    pp_12h = align_htf_to_ltf(prices, df_1d, pp_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate ATR(14) for volatility filter
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price touches S1 level with volume confirmation
            if price <= s1_12h[i] * 1.005 and vol_ratio > 1.5:  # Allow small buffer for touching
                signals[i] = size
                position = 1
            # Short: Price touches R1 level with volume confirmation
            elif price >= r1_12h[i] * 0.995 and vol_ratio > 1.5:  # Allow small buffer for touching
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price touches PP level or 2x ATR stop
            if price >= pp_12h[i] * 0.995 or price < low[i] - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price touches PP level or 2x ATR stop
            if price <= pp_12h[i] * 1.005 or price > high[i] + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1S1_Touch_Volume"
timeframe = "12h"
leverage = 1.0