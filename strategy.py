#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Weekly_Pivot_Reversal_With_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous 1w bar's pivot levels
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close[0] = close_1w[0]
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    
    # Pivot Point (PP) and key levels: R1, S1, R2, S2
    PP = (prev_high + prev_low + prev_close) / 3.0
    R1 = 2 * PP - prev_low
    S1 = 2 * PP - prev_high
    R2 = PP + (prev_high - prev_low)
    S2 = PP - (prev_high - prev_low)
    
    # Align weekly pivot levels to 6h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1w, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 1.5x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above S1 with volume spike and above weekly EMA50
            long_cond = (close[i] > S1_aligned[i] and 
                        close[i] <= S1_aligned[i-1] and  # crossed above S1 this bar
                        volume_spike[i] and
                        close[i] > ema50_aligned[i])
            
            # Short: Price crosses below R1 with volume spike and below weekly EMA50
            short_cond = (close[i] < R1_aligned[i] and 
                         close[i] >= R1_aligned[i-1] and  # crossed below R1 this bar
                         volume_spike[i] and
                         close[i] < ema50_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below S2 OR crosses below weekly EMA50
            if (close[i] < S2_aligned[i] and close[i] >= S2_aligned[i-1]) or \
               (close[i] < ema50_aligned[i] and close[i] >= ema50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above R2 OR crosses above weekly EMA50
            if (close[i] > R2_aligned[i] and close[i] <= R2_aligned[i-1]) or \
               (close[i] > ema50_aligned[i] and close[i] <= ema50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals