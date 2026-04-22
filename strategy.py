#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    # Load 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate weekly pivot points (R1, S1, R2, S2)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivots to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Calculate 60-period average volume for volume filter
    volume = prices['volume'].values
    vol_ma_60 = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or 
            np.isnan(vol_ma_60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        pivot_val = pivot_1w_aligned[i]
        r1_val = r1_1w_aligned[i]
        s1_val = s1_1w_aligned[i]
        r2_val = r2_1w_aligned[i]
        s2_val = s2_1w_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_60[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price > ema50_1d_val
        downtrend = price < ema50_1d_val
        
        # Volume filter: current volume > 1.5 * 60-period average volume
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above weekly R2 + 1d uptrend + volume spike
            if price > r2_val and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S2 + 1d downtrend + volume spike
            elif price < s2_val and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through opposite weekly S1/R1 or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below weekly S1 or volume drop
                if price < s1_val or not vol_spike:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above weekly R1 or volume drop
                if price > r1_val or not vol_spike:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_R2_S2_Breakout_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0