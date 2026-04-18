#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout
Hypothesis: 4-hour breakouts above upper or below lower Keltner Channel (20-period EMA ± 2.0*ATR) with volume confirmation and 1-day EMA34 trend filter.
Keltner Channels adapt to volatility, reducing false breakouts in ranging markets while capturing trends. Volume confirms institutional participation. EMA34 filter ensures alignment with higher timeframe trend. Designed for low trade frequency (target: 20-50/year) with strong performance in both bull and bear markets.
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
    
    # Calculate 1-day Keltner Channel
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day EMA20 for Keltner middle line
    ema20_1d = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        if i == 20:
            ema20_1d[i] = np.mean(close_1d[0:21])
        else:
            k = 2 / (20 + 1)
            ema20_1d[i] = close_1d[i] * k + ema20_1d[i-1] * (1 - k)
    
    # 1-day ATR for Keltner width
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with index 0
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(tr_1d)):
        if i == 14:
            atr_1d[i] = np.nanmean(tr_1d[1:15])  # first 14-period ATR
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14  # Wilder's smoothing
    
    # 1-day Keltner Bands
    upper_1d = ema20_1d + 2.0 * atr_1d
    lower_1d = ema20_1d - 2.0 * atr_1d
    
    # Align 1-day Keltner levels to 4h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 1-day EMA34 trend filter
    ema34_1d = np.full(len(close_1d), np.nan)
    for i in range(34, len(close_1d)):
        if i == 34:
            ema34_1d[i] = np.mean(close_1d[0:35])
        else:
            k = 2 / (34 + 1)
            ema34_1d[i] = close_1d[i] * k + ema34_1d[i-1] * (1 - k)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(ema20_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Keltner with volume spike and 1-day uptrend
            if (close[i] > upper_1d_aligned[i] and vol_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Keltner with volume spike and 1-day downtrend
            elif (close[i] < lower_1d_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below middle line or 1-day trend turns down
            if (close[i] < ema20_1d_aligned[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above middle line or 1-day trend turns up
            if (close[i] > ema20_1d_aligned[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Keltner_Channel_Breakout"
timeframe = "4h"
leverage = 1.0