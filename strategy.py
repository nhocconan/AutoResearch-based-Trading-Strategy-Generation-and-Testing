#!/usr/bin/env python3
# 4h_camarilla_pivot_volume_v10
# Hypothesis: Uses 1-day Camarilla pivot levels with volume confirmation and ADX trend filter.
# Target: 20-35 trades/year (80-140 total over 4 years) with strict entry conditions.
# Long when price breaks above H4 with volume and ADX>25; short when breaks below L4.
# Exit when price crosses H3/L3 or ADX<20. Uses discrete position sizing (0.25) to minimize churn.

import numpy as np
import pandas as pd
from mpt_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_volume_v10"
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
    
    # 1. 1-day Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    camarilla_h4 = np.zeros(len(df_1d))
    camarilla_l4 = np.zeros(len(df_1d))
    camarilla_h3 = np.zeros(len(df_1d))
    camarilla_l3 = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        pd_high = high_1d[i-1]
        pd_low = low_1d[i-1]
        pd_close = close_1d[i-1]
        
        pivot = (pd_high + pd_low + pd_close) / 3
        range_val = pd_high - pd_low
        
        camarilla_h4[i] = pivot + range_val * 1.1 / 2
        camarilla_l4[i] = pivot - range_val * 1.1 / 2
        camarilla_h3[i] = pivot + range_val * 1.1 / 4
        camarilla_l3[i] = pivot - range_val * 1.1 / 4
    
    # Align to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 2. Volume confirmation (20-period average)
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # 3. ADX trend filter (14-period) on 4h data
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (14-period)
    atr = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    if n >= 14:
        atr[13] = np.sum(tr[1:14])
        plus_dm_smooth[13] = np.sum(plus_dm[1:14])
        minus_dm_smooth[13] = np.sum(minus_dm[1:14])
        
        for i in range(14, n):
            atr[i] = atr[i-1] - (atr[i-1] / 14) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
    
    # Calculate DI and DX
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(14, n):
        if atr[i] != 0:
            plus_di[i] = 100 * (plus_dm_smooth[i] / atr[i])
            minus_di[i] = 100 * (minus_dm_smooth[i] / atr[i])
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX is smoothed DX
    adx = np.zeros(n)
    if n >= 28:
        adx[27] = np.sum(dx[14:28])
        for i in range(28, n):
            adx[i] = adx[i-1] - (adx[i-1] / 14) + dx[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(28, n):
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        vol_ok = volume[i] > vol_ma_20[i] * 1.8
        adx_ok = adx[i] > 25
        
        if position == 1:
            if close[i] < l3_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] > h3_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:
            if close[i] > h4_aligned[i] and vol_ok and adx_ok:
                position = 1
                signals[i] = 0.25
            elif close[i] < l4_aligned[i] and vol_ok and adx_ok:
                position = -1
                signals[i] = -0.25
    
    return signals