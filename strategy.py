#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_R4_R5_S4_S5_Breakout_1dTrend_VolumeFilter"
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
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily pivot points (R4, R5, S4, S5)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L
    r1_1d = (2 * pivot_1d) - low_1d
    # S1 = 2*P - H
    s1_1d = (2 * pivot_1d) - high_1d
    # R2 = P + (H - L)
    r2_1d = pivot_1d + (high_1d - low_1d)
    # S2 = P - (H - L)
    s2_1d = pivot_1d - (high_1d - low_1d)
    # R3 = H + 2*(P - L)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    # S3 = L - 2*(H - P)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    # R4 = R3 + (H - L)
    r4_1d = r3_1d + (high_1d - low_1d)
    # S4 = S3 - (H - L)
    s4_1d = s3_1d - (high_1d - low_1d)
    # R5 = R4 + (H - L)
    r5_1d = r4_1d + (high_1d - low_1d)
    # S5 = S4 - (H - L)
    s5_1d = s4_1d - (high_1d - low_1d)
    
    # Align daily pivots to 4h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r5_1d_aligned = align_htf_to_ltf(prices, df_1d, r5_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    s5_1d_aligned = align_htf_to_ltf(prices, df_1d, s5_1d)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(r5_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(s5_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above R4 + above daily EMA34 + volume confirmation
            if (close[i] > r4_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 1.5):
                # Avoid extreme extension beyond R5
                if close[i] <= r5_1d_aligned[i] * 1.02:
                    signals[i] = 0.25
                    position = 1
            # Short: price below S4 + below daily EMA34 + volume confirmation
            elif (close[i] < s4_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 1.5):
                # Avoid extreme extension beyond S5
                if close[i] >= s5_1d_aligned[i] * 0.98:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price below R4 OR below daily EMA34
            if close[i] < r4_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above S4 OR above daily EMA34
            if close[i] > s4_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals