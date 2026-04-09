#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    pp_1w = np.full(len(df_1w), np.nan)
    r1_1w = np.full(len(df_1w), np.nan)
    s1_1w = np.full(len(df_1w), np.nan)
    
    for i in range(1, len(df_1w)):
        ph = float(df_1w['high'].iloc[i-1])
        pl = float(df_1w['low'].iloc[i-1])
        pc = float(df_1w['close'].iloc[i-1])
        pp_1w[i] = (ph + pl + pc) / 3.0
        r1_1w[i] = pp_1w[i] + (ph - pl)
        s1_1w[i] = pp_1w[i] - (ph - pl)
    
    # Calculate daily Camarilla levels (using prior day's OHLC)
    r3_1d = np.full(len(df_1d), np.nan)
    s3_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        ph = float(df_1d['high'].iloc[i-1])
        pl = float(df_1d['low'].iloc[i-1])
        pc = float(df_1d['close'].iloc[i-1])
        r3_1d[i] = pc + (ph - pl) * 1.1 / 4
        s3_1d[i] = pc - (ph - pl) * 1.1 / 4
    
    # Align HTF values to 6h timeframe
    pp_1w_6h = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_6h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_6h = align_htf_to_ltf(prices, df_1w, s1_1w)
    r3_1d_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume confirmation: 3-period average (18h)
    vol_ma_3 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 3:
            vol_sum -= volume[i-3]
        if i >= 2:
            vol_ma_3[i] = vol_sum / 3
    
    # Calculate 14-period ADX for trend strength
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    tr_sum = 0.0
    for i in range(n):
        tr_sum += tr[i]
        if i >= 14:
            tr_sum -= tr[i-14]
        if i >= 13:
            atr[i] = tr_sum / 14
    
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    if len(atr) > 0 and not np.isnan(atr[-1]):
        atr_ma = np.full(n, np.nan)
        atr_sum = 0.0
        for i in range(n):
            if not np.isnan(atr[i]):
                atr_sum += atr[i]
                if i >= 14:
                    atr_sum -= atr[i-14]
                if i >= 13:
                    atr_ma[i] = atr_sum / 14
        
        for i in range(n):
            if not np.isnan(atr_ma[i]) and atr_ma[i] > 0:
                plus_di[i] = 100 * plus_dm[i] / atr_ma[i]
                minus_di[i] = 100 * minus_dm[i] / atr_ma[i]
                if (plus_di[i] + minus_di[i]) > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.full(n, np.nan)
    dx_sum = 0.0
    for i in range(n):
        if not np.isnan(dx[i]):
            dx_sum += dx[i]
            if i >= 14:
                dx_sum -= dx[i-14]
            if i >= 13:
                adx[i] = dx_sum / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(pp_1w_6h[i]) or 
            np.isnan(r1_1w_6h[i]) or 
            np.isnan(s1_1w_6h[i]) or 
            np.isnan(r3_1d_6h[i]) or 
            np.isnan(s3_1d_6h[i]) or 
            np.isnan(vol_ma_3[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly pivot OR ADX < 20 (trend weakening)
            if close[i] < pp_1w_6h[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly pivot OR ADX < 20
            if close[i] > pp_1w_6h[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price crosses above R1 weekly with volume confirmation AND ADX > 25
            vol_ratio = volume[i] / vol_ma_3[i] if vol_ma_3[i] > 0 else 0
            if (close[i] > r1_1w_6h[i] and 
                vol_ratio > 1.5 and 
                adx[i] > 25):
                position = 1
                signals[i] = 0.25
            # Enter short: price crosses below S1 weekly with volume confirmation AND ADX > 25
            elif (close[i] < s1_1w_6h[i] and 
                  vol_ratio > 1.5 and 
                  adx[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals