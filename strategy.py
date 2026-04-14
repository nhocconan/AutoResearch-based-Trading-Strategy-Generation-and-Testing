#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    camarilla_r4 = np.full(len(daily_high), np.nan)
    camarilla_r3 = np.full(len(daily_high), np.nan)
    camarilla_s3 = np.full(len(daily_high), np.nan)
    camarilla_s4 = np.full(len(daily_high), np.nan)
    
    for i in range(1, len(daily_high)):
        H = daily_high[i-1]
        L = daily_low[i-1]
        C = daily_close[i-1]
        diff = H - L
        
        camarilla_r4[i] = C + (diff * 1.1 / 2)
        camarilla_r3[i] = C + (diff * 1.1 / 4)
        camarilla_s3[i] = C - (diff * 1.1 / 4)
        camarilla_s4[i] = C - (diff * 1.1 / 2)
    
    # Volume filter: 20-period average
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / vol_count
            vol_sum -= volume[i-19]
            vol_count -= 1
    
    # Create arrays for alignment
    camarilla_r4_arr = camarilla_r4
    camarilla_r3_arr = camarilla_r3
    camarilla_s3_arr = camarilla_s3
    camarilla_s4_arr = camarilla_s4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if volume MA not ready
        if np.isnan(vol_ma[i]):
            continue
        
        # Get aligned Camarilla levels
        r4 = align_htf_to_ltf(prices, df_1d, camarilla_r4_arr)[i]
        r3 = align_htf_to_ltf(prices, df_1d, camarilla_r3_arr)[i]
        s3 = align_htf_to_ltf(prices, df_1d, camarilla_s3_arr)[i]
        s4 = align_htf_to_ltf(prices, df_1d, camarilla_s4_arr)[i]
        
        if np.isnan(r4) or np.isnan(r3) or np.isnan(s3) or np.isnan(s4):
            continue
        
        if position == 0:
            # Long: Break above R4 with volume
            if close[i] > r4 and volume[i] > vol_ma[i] * 1.5:
                position = 1
                signals[i] = position_size
            # Short: Break below S4 with volume
            elif close[i] < s4 and volume[i] > vol_ma[i] * 1.5:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price closes below R3
            if close[i] < r3:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price closes above S3
            if close[i] > s3:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_Breakout_Exit_Reversal_v1"
timeframe = "6h"
leverage = 1.0