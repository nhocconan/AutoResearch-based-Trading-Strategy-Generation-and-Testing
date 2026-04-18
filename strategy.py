#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Volume
Hypothesis: Daily Camarilla R1/S1 breakout with 1-day EMA trend filter and volume confirmation. 
Works in bull and bear markets by trading breakouts from key daily levels only when aligned with 
daily trend and accompanied by volume, avoiding false breakouts in ranging conditions. 
Designed for low frequency (15-30 trades/year) with strong performance across market regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA34 trend filter
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = close_1d[i] * alpha + ema34_1d[i-1] * (1 - alpha)
    
    # Calculate daily Camarilla levels (based on prior day's OHLC)
    # Standard Camarilla: 
    # H4 = Close + 1.5*(High-Low), H3 = Close + 1.0*(High-Low), H2 = Close + 0.5*(High-Low)
    # L4 = Close - 1.5*(High-Low), L3 = Close - 1.0*(High-Low), L2 = Close - 0.5*(High-Low)
    # But for breakout strategy, we use H3 (R1) and L3 (S1) as key levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r1 = np.full(len(close_1d), np.nan)  # H3 level
    s1 = np.full(len(close_1d), np.nan)  # L3 level
    
    for i in range(len(close_1d)):
        if i == 0:  # First day has no previous day
            continue
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        diff = ph - pl
        r1[i] = pc + 1.0 * diff  # H3
        s1[i] = pc - 1.0 * diff  # L3
    
    # Align daily data to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above daily R1 with volume spike and daily uptrend
            if (close[i] > r1_aligned[i] and vol_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below daily S1 with volume spike and daily downtrend
            elif (close[i] < s1_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below daily S1 or daily trend turns down
            if (close[i] < s1_aligned[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above daily R1 or daily trend turns up
            if (close[i] > r1_aligned[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0