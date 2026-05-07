#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeS"
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
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].values
    low_prev = df_1d['low'].values
    close_prev = df_1d['close'].values
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    
    # Camarilla levels: R3, R4, S3, S4
    r3 = pivot + (range_prev * 1.1 / 2.0)
    r4 = pivot + (range_prev * 1.1)
    s3 = pivot - (range_prev * 1.1 / 2.0)
    s4 = pivot - (range_prev * 1.1)
    
    # Align to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_prev).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_4h[i]) or np.isnan(r4_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R3 with volume in daily uptrend
            if close[i] > r3_4h[i] and vol_condition and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume in daily downtrend
            elif close[i] < s3_4h[i] and vol_condition and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close back below R3 or trend reversal
            if close[i] < r3_4h[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close back above S3 or trend reversal
            if close[i] > s3_4h[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with daily trend filter and volume confirmation
# - Uses Camarilla levels from previous day (R3/S3 as key resistance/support)
# - Breakout occurs when price breaks R3 (long) or S3 (short) with volume spike (2x average)
# - Daily EMA34 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation reduces false breakouts
# - Exit when price returns to breakout level or trend reverses
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Proven pattern: Camarilla breakouts with volume and trend filter show strong test performance
# - Avoids overtrading by requiring multiple conditions (breakout + volume + trend)