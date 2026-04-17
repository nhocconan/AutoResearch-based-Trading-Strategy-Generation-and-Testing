#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1d_Trend_v1
12h timeframe strategy using Donchian channel breakouts with 1d trend filter.
Long when price breaks above 20-period Donchian high and 1d EMA50 > EMA200.
Short when price breaks below 20-period Donchian low and 1d EMA50 < EMA200.
Exit when price crosses back through the Donchian midpoint.
Designed to capture trends with defined risk and limited trade frequency.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 12h Donchian Channel (20-period) ===
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_high[i] = np.max(high[i - lookback + 1:i + 1])
        donchian_low[i] = np.min(low[i - lookback + 1:i + 1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # === 1d Trend Filter: EMA50 and EMA200 ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMAs on 1d data
    ema50_1d = np.full(len(close_1d), np.nan)
    ema200_1d = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 50:
        alpha50 = 2 / (50 + 1)
        ema50_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema50_1d[i] = ema50_1d[i-1] + alpha50 * (close_1d[i] - ema50_1d[i-1])
    
    if len(close_1d) >= 200:
        alpha200 = 2 / (200 + 1)
        ema200_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema200_1d[i] = ema200_1d[i-1] + alpha200 * (close_1d[i] - ema200_1d[i-1])
    
    # Align 1d EMAs to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = max(200, 20)
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price breaks above Donchian high AND 1d EMA50 > EMA200 (uptrend)
            if (close[i] > donchian_high[i] and 
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below Donchian low AND 1d EMA50 < EMA200 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Price crosses below Donchian midpoint
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian midpoint
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1d_Trend_v1"
timeframe = "12h"
leverage = 1.0