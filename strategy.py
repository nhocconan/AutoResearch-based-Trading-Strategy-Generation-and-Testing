#!/usr/bin/env python3
"""
12h_RSI_Donchian_Breakout_Trend
Strategy: RSI(14) momentum + Donchian(20) breakout with trend filter
Timeframe: 12h
Trend filter: EMA(50) on 1d
Risk: Fixed position size 0.25, exits on trend reversal
Logic:
- Long: RSI > 55, price > 12h Donchian upper(20), price > 1d EMA(50)
- Short: RSI < 45, price < 12h Donchian lower(20), price < 1d EMA(50)
- Exit: Trend reversal (price crosses below/above 1d EMA(50))
Designed for low turnover and high conviction trades.
"""

name = "12h_RSI_Donchian_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1-day EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 12h Donchian channel (20)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high']
    low_12h = df_12h['low']
    donchian_upper = high_12h.rolling(window=20, min_periods=20).max()
    donchian_lower = low_12h.rolling(window=20, min_periods=20).min()
    
    donchian_upper_vals = donchian_upper.values
    donchian_lower_vals = donchian_lower.values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_vals)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_vals)
    
    # Calculate 12h RSI(14)
    close_12h = pd.Series(df_12h['close'])
    delta = close_12h.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup period
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI > 55, price > 12h Donchian upper, price > 1d EMA(50)
            if (rsi_aligned[i] > 55 and 
                close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI < 45, price < 12h Donchian lower, price < 1d EMA(50)
            elif (rsi_aligned[i] < 45 and 
                  close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal (price < 1d EMA(50))
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal (price > 1d EMA(50))
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals