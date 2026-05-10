#!/usr/bin/env python3
# 12h_1d_Camarilla_R3_S3_Breakout_With_Volume_Spike
# Hypothesis: Uses Camarilla levels from 1d timeframe to identify key support/resistance levels.
# Enters long when price breaks above R3 with volume spike (volume > 1.5x 20-period average).
# Enters short when price breaks below S3 with volume spike.
# Uses 1d ADX > 25 as trend filter to avoid false breakouts in ranging markets.
# Exits when price returns to the median (C) level or ADX drops below 20.
# Designed for 12h timeframe to target 15-30 trades/year, avoiding overtrading while capturing significant moves.

name = "12h_1d_Camarilla_R3_S3_Breakout_With_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla levels and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R3 = Close + 1.1*(High-Low)*1.1/2
    # S3 = Close - 1.1*(High-Low)*1.1/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 2
    camarilla_c = (high_1d + low_1d + close_1d) / 3  # Pivot point (median)
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_1d, camarilla_c)
    
    # 1d ADX for trend filter
    high_1d_arr = df_1d['high'].values
    low_1d_arr = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Calculate True Range and DM
    tr_1d = np.zeros(len(df_1d))
    plus_dm_1d = np.zeros(len(df_1d))
    minus_dm_1d = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        high_diff = high_1d_arr[i] - high_1d_arr[i-1]
        low_diff = low_1d_arr[i-1] - low_1d_arr[i]
        
        plus_dm_1d[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm_1d[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr_1d[i] = max(high_1d_arr[i] - low_1d_arr[i], 
                       abs(high_1d_arr[i] - close_1d_arr[i-1]), 
                       abs(low_1d_arr[i] - close_1d_arr[i-1]))
    
    # Smooth TR, +DM, -DM (Wilder's smoothing)
    atr_1d = np.zeros(len(df_1d))
    plus_dm_sum = np.zeros(len(df_1d))
    minus_dm_sum = np.zeros(len(df_1d))
    
    period = 14
    for i in range(1, len(df_1d)):
        if i < period:
            atr_1d[i] = np.mean(tr_1d[1:i+1]) if i >= 1 else 0
            plus_dm_sum[i] = np.sum(plus_dm_1d[1:i+1]) if i >= 1 else 0
            minus_dm_sum[i] = np.sum(minus_dm_1d[1:i+1]) if i >= 1 else 0
        else:
            atr_1d[i] = (atr_1d[i-1] * (period-1) + tr_1d[i]) / period
            plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period) + plus_dm_1d[i]
            minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period) + minus_dm_1d[i]
    
    # Calculate +DI and -DI
    plus_di_1d = np.zeros(len(df_1d))
    minus_di_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if atr_1d[i] != 0:
            plus_di_1d[i] = 100 * plus_dm_sum[i] / atr_1d[i]
            minus_di_1d[i] = 100 * minus_dm_sum[i] / atr_1d[i]
        else:
            plus_di_1d[i] = 0
            minus_di_1d[i] = 0
    
    # Calculate DX and ADX
    dx_1d = np.zeros(len(df_1d))
    adx_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        di_sum = plus_di_1d[i] + minus_di_1d[i]
        if di_sum != 0:
            dx_1d[i] = 100 * abs(plus_di_1d[i] - minus_di_1d[i]) / di_sum
        else:
            dx_1d[i] = 0
    
    # Smooth DX to get ADX
    for i in range(len(df_1d)):
        if i < 2*period:
            adx_1d[i] = np.mean(dx_1d[period:i+1]) if i >= period else 0
        else:
            adx_1d[i] = (adx_1d[i-1] * (period-1) + dx_1d[i]) / period
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike detection: current volume > 1.5x 20-period average
    vol_ma = np.zeros(n)
    vol_sum = 0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count > 20:
            vol_sum -= volume[i-20]
            vol_count = 20
        if vol_count >= 20:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = vol_sum / vol_count if vol_count > 0 else 0
    
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(adx_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_c_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and ADX > 25 (strong trend)
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i] and adx_1d_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike and ADX > 25 (strong trend)
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i] and adx_1d_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to pivot (C) level or ADX drops below 20 (weak trend)
            if close[i] < camarilla_c_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to pivot (C) level or ADX drops below 20 (weak trend)
            if close[i] > camarilla_c_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals