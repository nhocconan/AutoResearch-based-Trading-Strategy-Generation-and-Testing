#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (Standard)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r1_1w = pivot_1w + (range_1w * 1.0)
    s1_1w = pivot_1w - (range_1w * 1.0)
    r2_1w = pivot_1w + (range_1w * 2.0)
    s2_1w = pivot_1w - (range_1w * 2.0)
    r3_1w = pivot_1w + (range_1w * 3.0)
    s3_1w = pivot_1w - (range_1w * 3.0)
    r4_1w = pivot_1w + (range_1w * 4.0)
    s4_1w = pivot_1w - (range_1w * 4.0)
    
    # Align weekly pivot levels to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Get daily data for volume and volatility context
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily volume average for context
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 6h ATR for stop loss and position sizing
    tr_6h_1 = high - low
    tr_6h_2 = np.abs(high - np.roll(close, 1))
    tr_6h_3 = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr_6h_1, np.maximum(tr_6h_2, tr_6h_3))
    tr_6h[0] = tr_6h_1[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_1d_aligned[i] > np.nanmean(atr_1d_aligned[max(0, i-50):i+1])
        
        # Volume filter: current 6h volume > 1.5x daily average volume (scaled)
        vol_scaled = vol_ma_1d_aligned[i] * (6/24)  # scale daily volume to 6h equivalent
        volume_filter = volume[i] > vol_scaled * 1.5
        
        if position == 0:
            # Long: price breaks above weekly R4 with volume and volatility
            if close[i] > r4_1w_aligned[i] and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4 with volume and volatility
            elif close[i] < s4_1w_aligned[i] and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Trail stop: exit if price drops below 3*ATR from highest high since entry
            # Simplified: exit if price closes below weekly S3
            if close[i] < s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Trail stop: exit if price rises above 3*ATR from lowest low since entry
            # Simplified: exit if price closes above weekly R3
            if close[i] > r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_R3S3_R4S4_Breakout_Volume_Volatility"
timeframe = "6h"
leverage = 1.0