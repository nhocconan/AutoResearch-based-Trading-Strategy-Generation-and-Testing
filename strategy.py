#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly high, low, close for pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Calculate R1, S1, R2, S2
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Calculate daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume spike detection on 6h
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price array
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot = pivot_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        r2 = r2_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        ema200 = ema200_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        price = close[i]
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma
        
        # Trend filter: price above/below daily EMA200
        uptrend = price > ema200
        downtrend = price < ema200
        
        if position == 0:
            # Long: price crosses above weekly R1 with uptrend and volume
            if price > r1 and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below weekly S1 with downtrend and volume
            elif price < s1 and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to weekly pivot or volatility drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to weekly pivot or below
                if price <= pivot:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to weekly pivot or above
                if price >= pivot:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_EMA200_Volume"
timeframe = "6h"
leverage = 1.0