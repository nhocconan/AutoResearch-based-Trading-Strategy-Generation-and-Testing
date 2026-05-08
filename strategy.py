#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R3S4_Breakout_1dTrend_Volume"
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
    
    # Get daily data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume 20-period average
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate weekly pivot points (R3, S3, R4, S4)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = (2 * Pivot) - Low
    r1_1w = (2 * pivot_1w) - low_1w
    # S1 = (2 * Pivot) - High
    s1_1w = (2 * pivot_1w) - high_1w
    # R2 = Pivot + (High - Low)
    r2_1w = pivot_1w + (high_1w - low_1w)
    # S2 = Pivot - (High - Low)
    s2_1w = pivot_1w - (high_1w - low_1w)
    # R3 = High + 2*(Pivot - Low)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    # S3 = Low - 2*(High - Pivot)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    # R4 = R3 + (High - Low)
    r4_1w = r3_1w + (high_1w - low_1w)
    # S4 = S3 - (High - Low)
    s4_1w = s3_1w - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or
            np.isnan(s4_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x daily average volume
        vol_condition = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price crosses above R3 with volume and above daily EMA34
            if (close[i] > r3_1w_aligned[i] and 
                vol_condition and
                close[i] > ema_34_1d_aligned[i]):
                # Avoid extreme extension beyond R4
                if close[i] <= r4_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price crosses below S3 with volume and below daily EMA34
            elif (close[i] < s3_1w_aligned[i] and 
                  vol_condition and
                  close[i] < ema_34_1d_aligned[i]):
                # Avoid extreme extension beyond S4
                if close[i] >= s4_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price below R3 OR below daily EMA34
            if close[i] < r3_1w_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above S3 OR above daily EMA34
            if close[i] > s3_1w_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals