#!/usr/bin/env python3
name = "4h_Camarilla_Pivot_Reversion_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ===== 1d Trend Filter (HTF) =====
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # ===== 1d Camarilla Pivot Levels =====
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each day
    S1 = np.zeros_like(close_1d)
    S2 = np.zeros_like(close_1d)
    S3 = np.zeros_like(close_1d)
    R1 = np.zeros_like(close_1d)
    R2 = np.zeros_like(close_1d)
    R3 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Previous day's data
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d_prev[i]
        
        if not (np.isnan(H) or np.isnan(L) or np.isnan(C)):
            range_val = H - L
            if range_val > 0:
                S1[i] = C - (1.1 * range_val / 12)
                S2[i] = C - (1.1 * range_val / 6)
                S3[i] = C - (1.1 * range_val / 4)
                R1[i] = C + (1.1 * range_val / 12)
                R2[i] = C + (1.1 * range_val / 6)
                R3[i] = C + (1.1 * range_val / 4)
            else:
                S1[i] = S2[i] = S3[i] = R1[i] = R2[i] = R3[i] = C
        else:
            S1[i] = S2[i] = S3[i] = R1[i] = R2[i] = R3[i] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    
    # ===== 4h Volume Spike Filter =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or
            np.isnan(S1_4h[i]) or np.isnan(S2_4h[i]) or np.isnan(S3_4h[i]) or
            np.isnan(R1_4h[i]) or np.isnan(R2_4h[i]) or np.isnan(R3_4h[i]) or
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price touches S1/S2/S3 in uptrend with volume spike
            if (ema34_1d_aligned[i] > ema34_1d_aligned[i-1]) and vol_spike[i]:
                if (close[i] <= S1_4h[i] * 1.002 and close[i] >= S1_4h[i] * 0.998) or \
                   (close[i] <= S2_4h[i] * 1.002 and close[i] >= S2_4h[i] * 0.998) or \
                   (close[i] <= S3_4h[i] * 1.002 and close[i] >= S3_4h[i] * 0.998):
                    signals[i] = 0.25
                    position = 1
            # Short: Price touches R1/R2/R3 in downtrend with volume spike
            elif (ema34_1d_aligned[i] < ema34_1d_aligned[i-1]) and vol_spike[i]:
                if (close[i] >= R1_4h[i] * 0.998 and close[i] <= R1_4h[i] * 1.002) or \
                   (close[i] >= R2_4h[i] * 0.998 and close[i] <= R2_4h[i] * 1.002) or \
                   (close[i] >= R3_4h[i] * 0.998 and close[i] <= R3_4h[i] * 1.002):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Price touches R1 or trend breaks down
            if (close[i] >= R1_4h[i] * 0.998 and close[i] <= R1_4h[i] * 1.002) or \
               (ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price touches S1 or trend breaks up
            if (close[i] <= S1_4h[i] * 1.002 and close[i] >= S1_4h[i] * 0.998) or \
               (ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals