#!/usr/bin/env python3
# 12h_1w_Camarilla_R3_S3_Breakout_With_Volume_Spike
# Hypothesis: Uses weekly Camarilla pivot levels (R3/S3) as key support/resistance levels on 12h timeframe.
# Enters long when price breaks above R3 with volume spike, short when breaks below S3 with volume spike.
# Uses 1w ADX(14) to filter for strong trends (ADX > 25) to avoid false breakouts in ranging markets.
# Volume spike defined as current volume > 1.5 * 20-period average volume.
# Designed for 12-37 trades/year on 12h to avoid overtrading and work in both bull and bear markets.

name = "12h_1w_Camarilla_R3_S3_Breakout_With_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for Camarilla pivot levels and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from weekly data
    # Typical price = (H + L + C) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Pivot point = typical price
    pp = typical_price.values
    # R3 = PP + (H - L) * 1.1
    # S3 = PP - (H - L) * 1.1
    r3 = pp + (df_1w['high'] - df_1w['low']).values * 1.1
    s3 = pp - (df_1w['high'] - df_1w['low']).values * 1.1
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 1w ADX(14) for trend filter
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros(len(df_1w))
    minus_dm = np.zeros(len(df_1w))
    tr = np.zeros(len(df_1w))
    
    for i in range(1, len(df_1w)):
        high_diff = df_1w['high'].iloc[i] - df_1w['high'].iloc[i-1]
        low_diff = df_1w['low'].iloc[i-1] - df_1w['low'].iloc[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(
            df_1w['high'].iloc[i] - df_1w['low'].iloc[i],
            abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1]),
            abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1])
        )
    
    # Smooth TR, +DM, -DM using Wilder's smoothing
    atr_1w = np.zeros(len(df_1w))
    plus_dm_sum = np.zeros(len(df_1w))
    minus_dm_sum = np.zeros(len(df_1w))
    
    # Initial values
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[1:14])
        plus_dm_sum[13] = np.sum(plus_dm[1:14])
        minus_dm_sum[13] = np.sum(minus_dm[1:14])
        
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
            plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / 14) + plus_dm[i]
            minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / 14) + minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di_1w = np.zeros(len(df_1w))
    minus_di_1w = np.zeros(len(df_1w))
    
    for i in range(14, len(df_1w)):
        if atr_1w[i] != 0:
            plus_di_1w[i] = 100 * plus_dm_sum[i] / atr_1w[i]
            minus_di_1w[i] = 100 * minus_dm_sum[i] / atr_1w[i]
    
    # Calculate DX and ADX
    dx_1w = np.zeros(len(df_1w))
    adx_1w = np.zeros(len(df_1w))
    
    for i in range(14, len(df_1w)):
        di_sum = plus_di_1w[i] + minus_di_1w[i]
        if di_sum != 0:
            dx_1w[i] = 100 * abs(plus_di_1w[i] - minus_di_1w[i]) / di_sum
    
    # Smooth DX to get ADX
    if len(df_1w) >= 28:
        adx_1w[27] = np.mean(dx_1w[14:28])
        for i in range(28, len(df_1w)):
            adx_1w[i] = (adx_1w[i-1] * 13 + dx_1w[i]) / 14
    
    # Align ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume spike detection: current volume > 1.5 * 20-period average
    volume_ma = np.zeros(n)
    for i in range(20, n):
        volume_ma[i] = np.mean(volume[i-20:i])
    
    volume_spike = np.zeros(n, dtype=bool)
    for i in range(20, n):
        if volume_ma[i] > 0:
            volume_spike[i] = volume[i] > 1.5 * volume_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or i < 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and strong trend (ADX > 25)
            if close[i] > r3_aligned[i] and volume_spike[i] and adx_1w_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike and strong trend (ADX > 25)
            elif close[i] < s3_aligned[i] and volume_spike[i] and adx_1w_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below S3 or trend weakens (ADX < 20)
            if close[i] < s3_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above R3 or trend weakens (ADX < 20)
            if close[i] > r3_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals