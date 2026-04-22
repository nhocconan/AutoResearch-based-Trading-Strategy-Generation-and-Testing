#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Pivot levels and EMA200
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Pivot Point and key levels (S1, R1)
    pp_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    s1_1d = 2 * pp_1d - high_1d
    r1_1d = 2 * pp_1d - low_1d
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume spike filter (using 12h volume)
    volume = prices['volume'].values
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or 
            np.isnan(vol_ma_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_10[i]
        pp = pp_aligned[i]
        s1 = s1_aligned[i]
        r1 = r1_aligned[i]
        ema200 = ema200_aligned[i]
        
        # Volume filter: current volume > 2.0 * 10-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price crosses above S1 + volume spike + price > EMA200
            if price > s1 and vol_spike and price > ema200:
                signals[i] = 0.25
                position = 1
            # Short conditions: price crosses below R1 + volume spike + price < EMA200
            elif price < r1 and vol_spike and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through pivot point
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below pivot point
                if price < pp:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above pivot point
                if price > pp:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Pivot_S1_R1_Breakout_EMA200_Volume"
timeframe = "12h"
leverage = 1.0