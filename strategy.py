#!/usr/bin/env python3
name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    # 1w EMA26 trend filter
    ema_26_1w = pd.Series(df_1w['close']).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_26_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_26_1w)
    
    # 1d data for Camarilla pivot (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 4)   # R3 level
    s3 = pivot - (range_hl * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 1d timeframe (same timeframe, no alignment needed)
    r3_aligned = r3
    s3_aligned = s3
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 26  # Wait for EMA26 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_26_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 + above 1w EMA26 + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_26_1w_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 + below 1w EMA26 + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_26_1w_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Camarilla level or breaks in opposite direction
            if position == 1:
                if close[i] < s3_aligned[i] or close[i] < ema_26_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r3_aligned[i] or close[i] > ema_26_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals