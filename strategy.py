#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
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
    
    # Daily Camarilla pivot levels (from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close = df_1d['close'].values[:-1]
    prev_high = df_1d['high'].values[:-1]
    prev_low = df_1d['low'].values[:-1]
    
    # Calculate Camarilla levels for previous day
    R1 = prev_close + 0.1167 * (prev_high - prev_low)
    S1 = prev_close - 0.1167 * (prev_high - prev_low)
    R2 = prev_close + 0.2750 * (prev_high - prev_low)
    S2 = prev_close - 0.2750 * (prev_high - prev_low)
    R3 = prev_close + 0.4083 * (prev_high - prev_low)
    S3 = prev_close - 0.4083 * (prev_high - prev_low)
    
    # Align to 4h timeframe (previous day's levels available at start of day)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA34 trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike + in uptrend
            if (price > R1_4h[i] and 
                vol_spike[i] and 
                price > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S1 with volume spike + in downtrend
            elif (price < S1_4h[i] and 
                  vol_spike[i] and 
                  price < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price reaches R3 or trend fails
            if (price >= R3_4h[i] or 
                price < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S3 or trend fails
            if (price <= S3_4h[i] or 
                price > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals