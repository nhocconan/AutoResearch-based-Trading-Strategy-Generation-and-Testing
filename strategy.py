#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Camilla_Pivot_R1S1_Volume_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Daily pivot points from previous day
    prev_high_d = np.roll(df_1d['high'].values, 1)
    prev_low_d = np.roll(df_1d['low'].values, 1)
    prev_close_d = np.roll(df_1d['close'].values, 1)
    prev_high_d[0] = np.nan
    prev_low_d[0] = np.nan
    prev_close_d[0] = np.nan
    
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3.0
    r1_d = prev_close_d + (prev_high_d - prev_low_d) * 1.1 / 12.0
    s1_d = prev_close_d - (prev_high_d - prev_low_d) * 1.1 / 12.0
    
    # Weekly pivot points from previous week
    prev_high_w = np.roll(df_1w['high'].values, 1)
    prev_low_w = np.roll(df_1w['low'].values, 1)
    prev_close_w = np.roll(df_1w['close'].values, 1)
    prev_high_w[0] = np.nan
    prev_low_w[0] = np.nan
    prev_close_w[0] = np.nan
    
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3.0
    r1_w = prev_close_w + (prev_high_w - prev_low_w) * 1.1 / 12.0
    s1_w = prev_close_w - (prev_high_w - prev_low_w) * 1.1 / 12.0
    
    # Align to 12h timeframe
    pivot_d_12h = align_htf_to_ltf(prices, df_1d, pivot_d)
    r1_d_12h = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_12h = align_htf_to_ltf(prices, df_1d, s1_d)
    pivot_w_12h = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_12h = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_12h = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter: price above/below weekly pivot
    trend_up = close > pivot_w_12h
    trend_down = close < pivot_w_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_d_12h[i]) or np.isnan(r1_d_12h[i]) or np.isnan(s1_d_12h[i]) or \
           np.isnan(pivot_w_12h[i]) or np.isnan(r1_w_12h[i]) or np.isnan(s1_w_12h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume spike and weekly uptrend
            if price > r1_d_12h[i] and volume_spike and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with volume spike and weekly downtrend
            elif price < s1_d_12h[i] and volume_spike and trend_down[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below daily S1 (reversal signal)
            if price < s1_d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above daily R1 (reversal signal)
            if price > r1_d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals