#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_1dTrend
Hypothesis: Camarilla pivot levels from daily chart provide strong intraday support/resistance. 
Breakouts above R1 or below S1 with volume confirmation and aligned with 1-day EMA34 trend 
offer high-probability trades. Targets 15-30 trades/year by requiring confluence of 
price action, volume, and trend - reducing false breakouts in choppy markets.
Works in bull/bear via trend filter and avoids overtrading via strict entry conditions.
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
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    camarilla_r2 = np.full(len(close_1d), np.nan)
    camarilla_s2 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i == 0 or np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        rang = high_1d[i] - low_1d[i]
        camarilla_r1[i] = close_1d[i] + rang * 1.1 / 12
        camarilla_s1[i] = close_1d[i] - rang * 1.1 / 12
        camarilla_r2[i] = close_1d[i] + rang * 1.1 / 6
        camarilla_s2[i] = close_1d[i] - rang * 1.1 / 6
        camarilla_r3[i] = close_1d[i] + rang * 1.1 / 4
        camarilla_s3[i] = close_1d[i] - rang * 1.1 / 4
        camarilla_r4[i] = close_1d[i] + rang * 1.1 / 2
        camarilla_s4[i] = close_1d[i] - rang * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1-day EMA34 trend filter
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = close_1d[i] * alpha + ema34_1d[i-1] * (1 - alpha)
    
    # Align 1-day EMA34 to 6h timeframe
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(ema34_1d_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike and 1-day uptrend
            if (close[i] > r1_6h[i] and vol_spike[i] and 
                close[i] > ema34_1d_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and 1-day downtrend
            elif (close[i] < s1_6h[i] and vol_spike[i] and 
                  close[i] < ema34_1d_6h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below S1 or 1-day trend turns down
            if (close[i] < s1_6h[i] or close[i] < ema34_1d_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above R1 or 1-day trend turns up
            if (close[i] > r1_6h[i] or close[i] > ema34_1d_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0