#!/usr/bin/env python3
"""
4h_Parabolic_SAR_Trend_Follower
Hypothesis: Uses Parabolic SAR (0.02 step, 0.2 max) with EMA200 trend filter on 4h.
Enters long when PSAR flips below price and price > EMA200, short when PSAR flips above price and price < EMA200.
Uses 12h EMA34 as higher timeframe trend filter. Volume confirmation reduces false signals.
Designed for low trade frequency (~20-30/year) with strong trend capture in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Parabolic SAR parameters
    step = 0.02
    max_step = 0.2
    
    # Initialize SAR
    psar = np.full(n, np.nan)
    trend = np.full(n, np.nan)  # 1 for uptrend, -1 for downtrend
    ep = np.full(n, np.nan)     # extreme point
    af = np.full(n, np.nan)     # acceleration factor
    
    # Initialize first values
    if high[1] > high[0]:
        trend[1] = 1
        psar[1] = low[0]
        ep[1] = high[1]
        af[1] = step
    else:
        trend[1] = -1
        psar[1] = high[0]
        ep[1] = low[1]
        af[1] = step
    
    # Calculate PSAR
    for i in range(2, n):
        if trend[i-1] == 1:  # uptrend
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            # Ensure SAR doesn't exceed prior two lows
            if psar[i] > low[i-1]:
                psar[i] = low[i-1]
            if psar[i] > low[i-2]:
                psar[i] = low[i-2]
            
            # Check for reversal
            if low[i] < psar[i]:
                trend[i] = -1
                psar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = step
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + step, max_step)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # downtrend
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            # Ensure SAR doesn't go below prior two highs
            if psar[i] < high[i-1]:
                psar[i] = high[i-1]
            if psar[i] < high[i-2]:
                psar[i] = high[i-2]
            
            # Check for reversal
            if high[i] > psar[i]:
                trend[i] = 1
                psar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = step
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + step, max_step)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # EMA200 trend filter
    ema200 = np.full(n, np.nan)
    k = 2 / (200 + 1)
    for i in range(200, n):
        if i == 200:
            ema200[i] = np.mean(close[i-200+1:i+1])
        else:
            ema200[i] = close[i] * k + ema200[i-1] * (1 - k)
    
    # 12h EMA34 as higher timeframe trend filter
    df_12h = get_htf_data(prices, '12h')
    ema34_12h = np.full(len(df_12h), np.nan)
    k12h = 2 / (34 + 1)
    for i in range(34, len(df_12h)):
        if i == 34:
            ema34_12h[i] = np.mean(df_12h['close'].iloc[i-34+1:i+1])
        else:
            ema34_12h[i] = df_12h['close'].iloc[i] * k12h + ema34_12h[i-1] * (1 - k12h)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(psar[i]) or np.isnan(ema200[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: PSAR below price, price > EMA200, and 12h EMA34 rising
            if psar[i] < close[i] and close[i] > ema200[i] and ema34_12h_aligned[i] > ema34_12h_aligned[i-1] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: PSAR above price, price < EMA200, and 12h EMA34 falling
            elif psar[i] > close[i] and close[i] < ema200[i] and ema34_12h_aligned[i] < ema34_12h_aligned[i-1] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: PSAR flips above price or trend weakens
            if psar[i] > close[i] or ema200[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: PSAR flips below price or trend weakens
            if psar[i] < close[i] or ema200[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Parabolic_SAR_Trend_Follower"
timeframe = "4h"
leverage = 1.0