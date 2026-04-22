#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation.
Long when price breaks above 20-day high with 1-week EMA50 rising and volume spike.
Short when price breaks below 20-day low with 1-week EMA50 falling and volume spike.
Exit when price returns to the 20-day midpoint (mean reversion).
Designed for low trade frequency by requiring multiple confirmations and using daily price channels.
Works in both bull and bear markets by following the weekly trend.
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
    
    # Load 1-day data for Donchian channels and 1-week data for EMA50 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 50:
        return np.zeros(n)
    
    # 1-day Donchian(20) channels (use previous day's data to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period high and low using previous day's data
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    mid_20 = (high_max_20 + low_min_20) / 2.0
    
    # Align to 1-day timeframe (each day's levels apply to the entire day)
    high_max_20_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_1d, mid_20)
    
    # 1-week EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for EMA50
        # Skip if data not ready
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above 20-day high with 1-week EMA50 rising and volume spike
            if (close[i] > high_max_20_aligned[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with 1-week EMA50 falling and volume spike
            elif (close[i] < low_min_20_aligned[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to 20-day midpoint (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midpoint
                if close[i] < mid_20_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above midpoint
                if close[i] > mid_20_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_20_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0