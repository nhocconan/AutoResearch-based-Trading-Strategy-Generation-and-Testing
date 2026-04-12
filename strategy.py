#!/usr/bin/env python3
"""
6h_1w_1d_ElderRay_Pullback_v1
Hypothesis: On 6h timeframe, buy pullbacks to weekly EMA20 during bull regimes (weekly close > weekly EMA50) and sell pullbacks during bear regimes (weekly close < weekly EMA50). Uses Elder Ray (bull/bear power) to confirm momentum alignment. Designed for low trade frequency by requiring trend alignment and pullback to EMA20. Works in bull via long pullbacks in uptrends and in bear via short pullbacks in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_ElderRay_Pullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 and EMA50
    ema20_1w = np.zeros_like(close_1w)
    ema50_1w = np.zeros_like(close_1w)
    alpha20 = 2 / (20 + 1)
    alpha50 = 2 / (50 + 1)
    ema20_1w[0] = close_1w[0]
    ema50_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema20_1w[i] = alpha20 * close_1w[i] + (1 - alpha20) * ema20_1w[i-1]
        ema50_1w[i] = alpha50 * close_1w[i] + (1 - alpha50) * ema50_1w[i-1]
    
    # Weekly trend: bull if close > EMA50, bear if close < EMA50
    weekly_bull = close_1w > ema50_1w
    weekly_bear = close_1w < ema50_1w
    
    # Align weekly trend to 6m
    weekly_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_bull.astype(float))
    weekly_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_bear.astype(float))
    
    # === DAILY ELDER RAY ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA13 for Elder Ray
    ema13_1d = np.zeros_like(close_1d)
    alpha13 = 2 / (13 + 1)
    ema13_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema13_1d[i] = alpha13 * close_1d[i] + (1 - alpha13) * ema13_1d[i-1]
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray to 6m
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === 6H EMA20 FOR PULLBACK ENTRY ===
    ema20_6h = np.zeros(n)
    alpha20 = 2 / (20 + 1)
    ema20_6h[0] = close[0]
    for i in range(1, n):
        ema20_6h[i] = alpha20 * close[i] + (1 - alpha20) * ema20_6h[i-1]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(weekly_bull_aligned[i]) or np.isnan(weekly_bear_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema20_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: pullback to EMA20 with Elder Ray confirmation
        long_setup = (weekly_bull_aligned[i] > 0.5 and 
                     close[i] <= ema20_6h[i] * 1.005 and  # within 0.5% above EMA20
                     bull_power_aligned[i] > 0)  # bullish momentum
        
        short_setup = (weekly_bear_aligned[i] > 0.5 and 
                      close[i] >= ema20_6h[i] * 0.995 and  # within 0.5% below EMA20
                      bear_power_aligned[i] < 0)  # bearish momentum
        
        # Exit conditions: reverse when Elder Ray diverges or price extends too far
        exit_long = (weekly_bull_aligned[i] < 0.5 or 
                    bear_power_aligned[i] < 0 or  # bearish momentum appears
                    close[i] >= ema20_6h[i] * 1.02)  # extended 2% above EMA20
        
        exit_short = (weekly_bear_aligned[i] < 0.5 or 
                     bull_power_aligned[i] > 0 or  # bullish momentum appears
                     close[i] <= ema20_6h[i] * 0.98)  # extended 2% below EMA20
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals