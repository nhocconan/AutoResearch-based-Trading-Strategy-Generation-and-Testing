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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    weekly_r1 = 2 * weekly_pivot - low_1w
    weekly_s1 = 2 * weekly_pivot - high_1w
    weekly_r2 = weekly_pivot + (high_1w - low_1w)
    weekly_s2 = weekly_pivot - (high_1w - low_1w)
    weekly_r3 = high_1w + 2 * (weekly_pivot - low_1w)
    weekly_s3 = low_1w - 2 * (high_1w - weekly_pivot)
    
    # Align weekly pivot levels to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 50-period EMA on daily for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h volume spike (volume > 2.0x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30) + 1  # Ensure we have enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(weekly_r2_aligned[i]) or
            np.isnan(weekly_s2_aligned[i]) or
            np.isnan(weekly_r3_aligned[i]) or
            np.isnan(weekly_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike in uptrend
            if uptrend and vol_confirmed and close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume spike in downtrend
            elif downtrend and vol_confirmed and close[i] < weekly_s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below weekly pivot OR trend reverses
            if close[i] < weekly_pivot_aligned[i] or not uptrend:
                signals[i] = 0.0  # exit long
                position = 0
            else:
                signals[i] = 0.25  # hold long
        
        elif position == -1:
            # Short exit: price rises above weekly pivot OR trend reverses
            if close[i] > weekly_pivot_aligned[i] or not downtrend:
                signals[i] = 0.0  # exit short
                position = 0
            else:
                signals[i] = -0.25  # hold short
    
    return signals

name = "6h_WeeklyPivot_R1S1_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0