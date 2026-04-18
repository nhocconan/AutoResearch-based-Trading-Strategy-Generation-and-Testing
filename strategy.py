#!/usr/bin/env python3
"""
4h_Donchian_UpperLower_Breakout_1dTrend_Filter
Hypothesis: Buy when price breaks above 4-hour Donchian(20) upper band only if 1-day EMA50 is rising; sell when price breaks below lower band only if 1-day EMA50 is falling. Uses volume confirmation to avoid false breakouts. Designed to capture strong trending moves while avoiding whipsaws in ranging markets. Works in both bull and bear markets by following the daily trend direction.
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
    
    # 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # 1-day EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = close_1d[i] * alpha + ema50_1d[i-1] * (1 - alpha)
    ema50_1d_rising = np.diff(ema50_1d, prepend=np.nan) > 0
    
    # Align 1-day EMA50 and trend to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_rising.astype(float))
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirmed = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_1d_rising_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume and 1-day uptrend
            if (close[i] > donchian_high[i] and vol_confirmed[i] and 
                ema50_1d_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and 1-day downtrend
            elif (close[i] < donchian_low[i] and vol_confirmed[i] and 
                  not ema50_1d_rising_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian low or trend turns down
            if (close[i] < donchian_low[i] or not ema50_1d_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high or trend turns up
            if (close[i] > donchian_high[i] or ema50_1d_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_UpperLower_Breakout_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0