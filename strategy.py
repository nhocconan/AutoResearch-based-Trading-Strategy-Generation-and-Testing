#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_12hEMA34
Long: Close > R1 + Volume spike + 12h EMA34 rising
Short: Close < S1 + Volume spike + 12h EMA34 falling
Exit: Opposite signal or price crosses H3/L3 (wider exit)
Uses Camarilla pivot levels from daily + volume confirmation + 12h trend filter.
Designed to work in both bull and bear markets by filtering with 12h EMA trend.
Target: 75-200 total trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1, H3, L3
    # R1 = Close + (High - Low) * 1.12 / 12
    # S1 = Close - (High - Low) * 1.12 / 12
    # H3 = Close + (High - Low) * 1.12 / 4
    # L3 = Close - (High - Low) * 1.12 / 4
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.12 / 12
    s1 = close_1d - range_1d * 1.12 / 12
    h3 = close_1d + range_1d * 1.12 / 4
    l3 = close_1d - range_1d * 1.12 / 4
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate volume spike: volume > 2x 20-period SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(34, 20)  # need EMA34 and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        ema_34_val = ema_34_12h_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        
        if position == 0:
            # Long: Close > R1 + Volume spike + 12h EMA34 rising
            if price > r1_val and vol > 2.0 * vol_sma_val and ema_34_val > ema_34_12h[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 + Volume spike + 12h EMA34 falling
            elif price < s1_val and vol > 2.0 * vol_sma_val and ema_34_val < ema_34_12h[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close < H3 (wider exit) or 12h EMA34 turns down
            if price < h3_val or ema_34_val < ema_34_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close > L3 (wider exit) or 12h EMA34 turns up
            if price > l3_val or ema_34_val > ema_34_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_12hEMA34"
timeframe = "4h"
leverage = 1.0