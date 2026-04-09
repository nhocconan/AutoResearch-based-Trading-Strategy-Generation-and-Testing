#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 1-day ATR breakout with volume confirmation and ADX trend filter
# Works in bull/bear by capturing breakouts with volume surge, filtering by ADX>25 for trending markets
# Target: 20-40 trades/year to avoid fee drag, focusing on high-probability breakouts with trend confirmation

name = "4h_1d_atr_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 1:
            continue
        tr1 = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        tr2 = abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1])
        tr3 = abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        tr = max(tr1, tr2, tr3)
        if i < 14:
            if i == 0:
                atr_1d[i] = tr
            else:
                atr_1d[i] = (atr_1d[i-1] * i + tr) / (i + 1)
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14
    
    # Align 1d ATR to 4h timeframe
    atr_1d_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily ADX(14) for trend filter
    # First calculate +DM and -DM
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    tr_list = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        high_diff = df_1d['high'].iloc[i] - df_1d['high'].iloc[i-1]
        low_diff = df_1d['low'].iloc[i-1] - df_1d['low'].iloc[i]
        
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        
        tr1 = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        tr2 = abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1])
        tr3 = abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        tr_list[i] = max(tr1, tr2, tr3)
    
    # Smooth +DM, -DM, and TR
    atr_smoothed = np.zeros(len(df_1d))
    plus_dm_smoothed = np.zeros(len(df_1d))
    minus_dm_smoothed = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i < 14:
            if i == 0:
                atr_smoothed[i] = tr_list[i]
                plus_dm_smoothed[i] = plus_dm[i]
                minus_dm_smoothed[i] = minus_dm[i]
            else:
                atr_smoothed[i] = (atr_smoothed[i-1] * i + tr_list[i]) / (i + 1)
                plus_dm_smoothed[i] = (plus_dm_smoothed[i-1] * i + plus_dm[i]) / (i + 1)
                minus_dm_smoothed[i] = (minus_dm_smoothed[i-1] * i + minus_dm[i]) / (i + 1)
        else:
            atr_smoothed[i] = (atr_smoothed[i-1] * 13 + tr_list[i]) / 14
            plus_dm_smoothed[i] = (plus_dm_smoothed[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_smoothed[i] = (minus_dm_smoothed[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate DI+ and DI-
    plus_di = np.zeros(len(df_1d))
    minus_di = np.zeros(len(df_1d))
    dx = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if atr_smoothed[i] > 0:
            plus_di[i] = 100 * plus_dm_smoothed[i] / atr_smoothed[i]
            minus_di[i] = 100 * minus_dm_smoothed[i] / atr_smoothed[i]
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Calculate ADX (smoothed DX)
    adx_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 14:
            continue
        if i == 14:
            adx_1d[i] = np.mean(dx[1:15])  # Average of first 14 DX values
        else:
            adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # Align 1d ADX to 4h timeframe
    adx_1d_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 4-period average (16h)
    vol_ma_4 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 4:
            vol_sum -= volume[i-4]
        if i >= 3:
            vol_ma_4[i] = vol_sum / 4
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_4h[i]) or 
            np.isnan(adx_1d_4h[i]) or 
            np.isnan(vol_ma_4[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below entry point - ATR or ADX drops below 20 (trend weakening)
            if (close[i] < close[i-1] - atr_1d_4h[i] * 0.5) or adx_1d_4h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above entry point + ATR or ADX drops below 20
            if (close[i] > close[i-1] + atr_1d_4h[i] * 0.5) or adx_1d_4h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above previous close + ATR with volume confirmation AND ADX > 25 (trending)
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if (close[i] > close[i-1] + atr_1d_4h[i] and 
                vol_ratio > 2.0 and 
                adx_1d_4h[i] > 25):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below previous close - ATR with volume confirmation AND ADX > 25
            elif (close[i] < close[i-1] - atr_1d_4h[i] and 
                  vol_ratio > 2.0 and 
                  adx_1d_4h[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals