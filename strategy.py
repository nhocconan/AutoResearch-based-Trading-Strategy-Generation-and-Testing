#!/usr/bin/env python3
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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points and support/resistance levels
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R1 = 2*PP - Low, S1 = 2*PP - High
    # R2 = PP + (High - Low), S2 = PP - (High - Low)
    # R3 = High + 2*(PP - Low), S3 = Low - 2*(High - PP)
    # R4 = 3*PP - 2*Low, S4 = 3*PP - 2*High
    
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    r3 = high_1d + 2 * (pp - low_1d)
    s3 = low_1d - 2 * (high_1d - pp)
    r4 = 3 * pp - 2 * low_1d
    s4 = 3 * pp - 2 * high_1d
    
    # Align daily pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 6-period RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(6, n):
        if i == 6:
            avg_gain[i] = np.mean(gain[1:7])
            avg_loss[i] = np.mean(loss[1:7])
        else:
            avg_gain[i] = (avg_gain[i-1] * 5 + gain[i]) / 6
            avg_loss[i] = (avg_loss[i-1] * 5 + loss[i]) / 6
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_6 = np.full(n, np.nan)
    rsi_6[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = 6
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(rsi_6[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price near S3/S4 support with oversold RSI
            if ((price <= s3_aligned[i] * 1.02 or price <= s4_aligned[i] * 1.02) and 
                rsi_6[i] < 30):
                signals[i] = 0.25
                position = 1
            # Short: price near R3/R4 resistance with overbought RSI
            elif ((price >= r3_aligned[i] * 0.98 or price >= r4_aligned[i] * 0.98) and 
                  rsi_6[i] > 70):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price near resistance or RSI overbought
            if (price >= r1_aligned[i] * 0.98 or 
                rsi_6[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price near support or RSI oversold
            if (price <= s1_aligned[i] * 1.02 or 
                rsi_6[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_S3S4_RSI6_Reversion_v1"
timeframe = "6h"
leverage = 1.0