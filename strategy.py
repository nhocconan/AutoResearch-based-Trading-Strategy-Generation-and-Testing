#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_1wTrend_Volume
Hypothesis: Daily breakouts above/below Keltner Channel (ATR-based) with weekly EMA trend filter and volume confirmation.
Keltner Channels adapt to volatility, providing dynamic support/resistance. Weekly EMA ensures trend alignment.
Designed for low trade frequency (target: 10-25/year) with strong performance in both bull and bear markets via volatility-adjusted breakouts.
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
    
    # Calculate weekly EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 with proper smoothing
    ema20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema20_1w[i] = close_1w[i] * alpha + ema20_1w[i-1] * (1 - alpha)
    
    # Align weekly EMA20 to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate ATR(10) for Keltner Channel
    atr = np.full(n, np.nan)
    if n >= 10:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # ATR with Wilder's smoothing
        atr[9] = np.mean(tr[0:10])
        for i in range(10, n):
            atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # Calculate Keltner Channel (20 EMA ± 2 * ATR)
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = close[i] * alpha + ema20[i-1] * (1 - alpha)
    
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    # Volume spike: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Keltner upper with volume spike and weekly uptrend
            if (close[i] > kc_upper[i] and vol_spike[i] and 
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Keltner lower with volume spike and weekly downtrend
            elif (close[i] < kc_lower[i] and vol_spike[i] and 
                  close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Keltner lower or weekly trend turns down
            if (close[i] < kc_lower[i] or close[i] < ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Keltner upper or weekly trend turns up
            if (close[i] > kc_upper[i] or close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Keltner_Channel_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0