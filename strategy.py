#!/usr/bin/env python3
"""
12h_1d_KAMA_Trend_Channel_Breakout
Hypothesis: Use 12h KAMA trend + 12h Donchian breakout with 1d volume confirmation.
KAMA adapts to noise, reducing whipsaws. Breakout beyond 12h Donchian(10) captures momentum.
Volume filter ensures participation. Works in bull (breakouts above upper band) and bear
(breakdowns below lower band) by requiring KAMA alignment. Targets 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for KAMA and Donchian
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h KAMA (adaptive trend)
    def calculate_kama(close_arr, period=10):
        change = np.abs(np.diff(close_arr, n=period))
        # Volatility: sum of absolute changes over 'period' periods
        volatility = np.zeros_like(close_arr)
        for i in range(len(close_arr)):
            if i >= period:
                volatility[i] = np.sum(np.abs(np.diff(close_arr[i-period+1:i+1], n=1)))
        er = np.zeros_like(close_arr)
        mask = volatility != 0
        er[mask] = change[mask] / volatility[mask]
        # Smoothing constants
        sc = (er * (0.665 - 0.0645) + 0.0645) ** 2
        kama = np.full_like(close_arr, np.nan)
        if len(close_arr) > period:
            kama[period-1] = close_arr[period-1]
            for i in range(period, len(close_arr)):
                kama[i] = kama[i-1] + sc[i] * (close_arr[i] - kama[i-1])
        return kama
    
    kama_12h = calculate_kama(close_12h, 10)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 12h Donchian channels (10-period)
    def calculate_donchian(high_arr, low_arr, period=10):
        upper = np.full_like(high_arr, np.nan)
        lower = np.full_like(low_arr, np.nan)
        for i in range(len(high_arr)):
            if i >= period - 1:
                upper[i] = np.max(high_arr[i-period+1:i+1])
                lower[i] = np.min(low_arr[i-period+1:i+1])
        return upper, lower
    
    donch_up_12h, donch_dn_12h = calculate_donchian(high_12h, low_12h, 10)
    donch_up_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_up_12h)
    donch_dn_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_dn_12h)
    
    # Get 1d volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    # 1d volume MA (20-period)
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_12h_aligned[i]) or 
            np.isnan(donch_up_12h_aligned[i]) or 
            np.isnan(donch_dn_12h_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]  # current 12h volume > 1d avg volume
        
        if position == 0:
            # Long: price breaks above Donchian upper, with volume, and KAMA uptrend
            if (close[i] > donch_up_12h_aligned[i] and 
                vol_confirm and 
                close[i] > kama_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, with volume, and KAMA downtrend
            elif (close[i] < donch_dn_12h_aligned[i] and 
                  vol_confirm and 
                  close[i] < kama_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below KAMA or breaks below Donchian lower
            if (close[i] < kama_12h_aligned[i] or 
                close[i] < donch_dn_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above KAMA or breaks above Donchian upper
            if (close[i] > kama_12h_aligned[i] or 
                close[i] > donch_up_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_KAMA_Trend_Channel_Breakout"
timeframe = "12h"
leverage = 1.0