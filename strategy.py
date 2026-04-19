# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Strategy: 4h_1d_Camarilla_R1S1_Breakout_Volume_Divergence
Timeframe: 4h
Hypothesis: Camarilla R1/S1 breakouts with volume divergence (bullish/bearish) and 1d trend filter work in both bull and bear markets.
Volume divergence: price makes new high/low but volume fails to confirm, signaling exhaustion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_R1S1_Breakout_Volume_Divergence"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot levels from previous day
    prev_close_d = np.roll(close_1d, 1)
    prev_close_d[0] = np.nan
    prev_high_d = np.roll(high_1d, 1)
    prev_high_d[0] = np.nan
    prev_low_d = np.roll(low_1d, 1)
    prev_low_d[0] = np.nan
    
    # Camarilla levels
    # R4 = C + (H-L)*1.1/2
    r4_d = prev_close_d + (prev_high_d - prev_low_d) * 1.1 / 2.0
    # R3 = C + (H-L)*1.1/4
    r3_d = prev_close_d + (prev_high_d - prev_low_d) * 1.1 / 4.0
    # R2 = C + (H-L)*1.1/6
    r2_d = prev_close_d + (prev_high_d - prev_low_d) * 1.1 / 6.0
    # R1 = C + (H-L)*1.1/12
    r1_d = prev_close_d + (prev_high_d - prev_low_d) * 1.1 / 12.0
    # S1 = C - (H-L)*1.1/12
    s1_d = prev_close_d - (prev_high_d - prev_low_d) * 1.1 / 12.0
    # S2 = C - (H-L)*1.1/6
    s2_d = prev_close_d - (prev_high_d - prev_low_d) * 1.1 / 6.0
    # S3 = C - (H-L)*1.1/4
    s3_d = prev_close_d - (prev_high_d - prev_low_d) * 1.1 / 4.0
    # S4 = C - (H-L)*1.1/2
    s4_d = prev_close_d - (prev_high_d - prev_low_d) * 1.1 / 2.0
    
    # Align to 4h timeframe
    r1_d_4h = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_4h = align_htf_to_ltf(prices, df_1d, s1_d)
    r2_d_4h = align_htf_to_ltf(prices, df_1d, r2_d)
    s2_d_4h = align_htf_to_ltf(prices, df_1d, s2_d)
    r3_d_4h = align_htf_to_ltf(prices, df_1d, r3_d)
    s3_d_4h = align_htf_to_ltf(prices, df_1d, s3_d)
    r4_d_4h = align_htf_to_ltf(prices, df_1d, r4_d)
    s4_d_4h = align_htf_to_ltf(prices, df_1d, s4_d)
    
    # Volume divergence: compare current volume to recent volume trend
    # Bullish divergence: price makes lower low but volume is higher than recent avg
    # Bearish divergence: price makes higher high but volume is lower than recent avg
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Price momentum for divergence detection
    price_change_5 = pd.Series(close).pct_change(5).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(r1_d_4h[i]) or np.isnan(s1_d_4h[i]) or np.isnan(r2_d_4h[i]) or \
           np.isnan(s2_d_4h[i]) or np.isnan(r3_d_4h[i]) or np.isnan(s3_d_4h[i]) or \
           np.isnan(r4_d_4h[i]) or np.isnan(s4_d_4h[i]) or np.isnan(vol_ma_20[i]) or \
           np.isnan(vol_ma_50[i]) or np.isnan(price_change_5[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        vol_ma_long = vol_ma_50[i]
        price_mom = price_change_5[i]
        
        # Volume divergence signals
        # Bullish divergence: price down but volume up (accumulation)
        bullish_div = price_mom < -0.01 and vol > vol_ma_long * 1.3
        # Bearish divergence: price up but volume down (distribution)
        bearish_div = price_mom > 0.01 and vol < vol_ma_long * 0.7
        
        if position == 0:
            # Long: Price near S1/S2 with bullish volume divergence
            if (price <= s1_d_4h[i] * 1.005 or price <= s2_d_4h[i] * 1.005) and bullish_div:
                signals[i] = 0.25
                position = 1
            # Short: Price near R1/R2 with bearish volume divergence
            elif (price >= r1_d_4h[i] * 0.995 or price >= r2_d_4h[i] * 0.995) and bearish_div:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price reaches R2 or bearish divergence appears
            if price >= r2_d_4h[i] or bearish_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price reaches S2 or bullish divergence appears
            if price <= s2_d_4h[i] or bullish_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals