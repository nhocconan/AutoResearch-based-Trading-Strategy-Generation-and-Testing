#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Pivot_R1S1_Breakout_VolumeATR"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for pivot points and ATR
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ATR(14)
    tr1 = np.maximum(high_4h[1:], close_4h[:-1]) - np.minimum(low_4h[1:], close_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 4h pivot points: P = (H+L+C)/3
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    s1_4h = 2 * pivot_4h - high_4h
    r1_4h = 2 * pivot_4h - low_4h
    
    # Align to 1h timeframe
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 1.5x 24-period average (1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(r1_4h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        atr = atr_4h_aligned[i]
        ema = ema_1d_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        s1 = s1_4h_aligned[i]
        r1 = r1_4h_aligned[i]
        
        if position == 0:
            # Long: Break above R1 with volume and above 1d EMA50
            if price > r1 and volume_confirmed and price > ema:
                signals[i] = 0.20
                position = 1
            # Short: Break below S1 with volume and below 1d EMA50
            elif price < s1 and volume_confirmed and price < ema:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price closes below S1 or ATR stop (2.0x ATR)
            if price < s1 or price < (high[i] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price closes above R1 or ATR stop (2.0x ATR)
            if price > r1 or price > (low[i] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals