#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Pivot_R1S1_Breakout_VolumeATR_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend context (once before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get daily data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: P = (H+L+C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 1h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 4h ATR for volatility filter (14-period)
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], np.abs(high_4h[1:] - close_4h[:-1]))
    tr_4h = np.maximum(tr_4h, np.abs(low_4h[1:] - close_4h[:-1]))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # 4h EMA200 for trend filter
    ema_200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Volume confirmation: current volume > 2.0x 20-period average (1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        atr = atr_14_4h_aligned[i]
        ema200 = ema_200_4h_aligned[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        # Determine trend from 4h EMA200
        uptrend = price > ema200
        downtrend = price < ema200
        
        if position == 0:
            # Long: break above R1 with volume in uptrend
            if uptrend and price > r1 and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 with volume in downtrend
            elif downtrend and price < s1 and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price below pivot or ATR-based stop
            if price < pivot or price < close[i-1] - 2.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price above pivot or ATR-based stop
            if price > pivot or price > close[i-1] + 2.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals