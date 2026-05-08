#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_PivotBreakout_1dTrend_Volume"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily data for Pivot Points (PP, R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n1d = len(close_1d)
    pivot_pp = np.full(n1d, np.nan)
    pivot_r1 = np.full(n1d, np.nan)
    pivot_s1 = np.full(n1d, np.nan)
    
    for i in range(1, n1d):
        PH = high_1d[i-1]
        PL = low_1d[i-1]
        PC = close_1d[i-1]
        
        PP = (PH + PL + PC) / 3.0
        R1 = 2 * PP - PL
        S1 = 2 * PP - PH
        
        pivot_pp[i] = PP
        pivot_r1[i] = R1
        pivot_s1[i] = S1
    
    # Align Pivot levels to 12h timeframe
    pivot_pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_pp)
    pivot_r1_aligned = align_htf_to_ltf(prices, df_1d, pivot_r1)
    pivot_s1_aligned = align_htf_to_ltf(prices, df_1d, pivot_s1)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_pp_aligned[i]) or np.isnan(pivot_r1_aligned[i]) or 
            np.isnan(pivot_s1_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 1d uptrend + volume spike
            long_cond = (close[i] > pivot_r1_aligned[i] and 
                        ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below S1 with 1d downtrend + volume spike
            short_cond = (close[i] < pivot_s1_aligned[i] and 
                         ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below PP (reversion to mean)
            if close[i] < pivot_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above PP (reversion to mean)
            if close[i] > pivot_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals