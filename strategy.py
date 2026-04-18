#!/usr/bin/env python3
"""
4h_MultiTF_Donchian_Breakout_Signal_With_Adaptive_Exit
Hypothesis: Uses 4h Donchian breakout with 1d volume confirmation and 1w EMA trend filter. 
Exits when price crosses back through Donchian midline or volatility collapses. 
Designed for low trade frequency (<50/year) to minimize fee drag while capturing strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 1d average volume (30-period)
    vol_30 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 30:
        vol_series = pd.Series(df_1d['volume'].values)
        vol_30 = vol_series.rolling(window=30, min_periods=30).mean().values
    
    # Align 1d average volume to 4h timeframe
    vol_30_aligned = align_htf_to_ltf(prices, df_1d, vol_30)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_series = pd.Series(close_1w)
        ema_50_1w = ema_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian midline for exit signal
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: current volume > 1.5x 1d average volume
    vol_spike = volume > (vol_30_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for Donchian and EMAs
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_30_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and above 1w EMA50
            if close[i] > donchian_high[i] and vol_spike[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike and below 1w EMA50
            elif close[i] < donchian_low[i] and vol_spike[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midline OR volatility drops
            if close[i] < donchian_mid[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midline OR volatility drops
            if close[i] > donchian_mid[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_MultiTF_Donchian_Breakout_Signal_With_Adaptive_Exit"
timeframe = "4h"
leverage = 1.0