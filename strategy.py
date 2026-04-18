#!/usr/bin/env python3
"""
1d_SR_Channel_Breakout_Volume_1wTrend
Hypothesis: Daily breakouts above weekly channel resistance or below support with weekly trend filter and volume confirmation.
Targets low frequency (~10-20 trades/year) to avoid fee drag, works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly high/low/close for channel
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly 20-period EMA for trend filter
    ema20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema20_1w[i] = close_1w[i] * alpha + ema20_1w[i-1] * (1 - alpha)
    
    # Align weekly channel and trend to daily
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume spike: current volume > 2.0 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Weekly EMA20 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_1w_aligned[i]) or np.isnan(low_1w_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly high with volume spike and weekly uptrend
            if (close[i] > high_1w_aligned[i] and vol_spike[i] and 
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly low with volume spike and weekly downtrend
            elif (close[i] < low_1w_aligned[i] and vol_spike[i] and 
                  close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly low or weekly trend turns down
            if (close[i] < low_1w_aligned[i] or close[i] < ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly high or weekly trend turns up
            if (close[i] > high_1w_aligned[i] or close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_SR_Channel_Breakout_Volume_1wTrend"
timeframe = "1d"
leverage = 1.0