#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_v2
# Hypothesis: Uses weekly Camarilla pivot levels on daily chart. Long when price closes above L4 with volume > 1.5x average; short when price closes below H4 with volume > 1.5x average. Includes ADX trend filter to avoid choppy markets. Designed to work in both bull and bear markets by fading overextensions at key levels. Target: 10-25 trades/year (40-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ADX(14) for trend filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]
    
    # +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        else:
            plus_dm[i] = 0
        if down > up and down > 0:
            minus_dm[i] = down
        else:
            minus_dm[i] = 0
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    tr_smooth = np.zeros(n)
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    tr_smooth[0] = tr[0]
    for i in range(1, n):
        plus_dm_smooth[i] = 0.9 * plus_dm_smooth[i-1] + 0.1 * plus_dm[i]
        minus_dm_smooth[i] = 0.9 * minus_dm_smooth[i-1] + 0.1 * minus_dm[i]
        tr_smooth[i] = 0.9 * tr_smooth[i-1] + 0.1 * tr[i]
    
    # +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(n):
        if tr_smooth[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    # DX and ADX
    dx = np.zeros(n)
    for i in range(n):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    adx = np.zeros(n)
    adx[0] = dx[0]
    for i in range(1, n):
        adx[i] = 0.9 * adx[i-1] + 0.1 * dx[i]
    
    # Load weekly data ONCE before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous week
    ph = df_1w['high'].values
    pl = df_1w['low'].values
    pc = df_1w['close'].values
    range_1w = ph - pl
    h4 = pc + 1.5 * range_1w * 1.1 / 2
    l4 = pc - 1.5 * range_1w * 1.1 / 2
    h3 = pc + 1.25 * range_1w * 1.1 / 2
    l3 = pc - 1.25 * range_1w * 1.1 / 2
    
    # Align Camarilla levels to daily timeframe (wait for previous week's close)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(20, n):
        if np.isnan(adx[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade in trending markets (ADX > 20)
        trend_filter = adx[i] > 20
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:
            if close[i] < l3_aligned[i] and close[i-1] >= l3_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] > h3_aligned[i] and close[i-1] <= h3_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:
            if close[i] > l4_aligned[i] and close[i-1] <= l4_aligned[i-1] and vol_ok and trend_filter:
                position = 1
                signals[i] = 0.25
            elif close[i] < h4_aligned[i] and close[i-1] >= h4_aligned[i-1] and vol_ok and trend_filter:
                position = -1
                signals[i] = -0.25
    
    return signals