#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian breakout with 1-week EMA20 trend and volume spike.
Long when price breaks above 20-day high with 1-week EMA20 rising and volume spike.
Short when price breaks below 20-day low with 1-week EMA20 falling and volume spike.
Exit when price retests the midpoint of the Donchian channel.
Donchian channels provide trend-following structure; 1-week EMA20 filters trend direction;
volume spike confirms institutional participation. Designed for low trade frequency by requiring
multiple confirmations and using daily price levels. Works in both bull and bear markets
by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Donchian channel - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channel (20-day high/low)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Align to 1d timeframe (each day's levels apply to the entire day)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Load 1-week data for EMA20 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after enough data for EMA20 and Donchian
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above 20-day high with 1-week EMA20 rising and volume spike
            if (close[i] > high_20_aligned[i] and 
                ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with 1-week EMA20 falling and volume spike
            elif (close[i] < low_20_aligned[i] and 
                  ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price retests midpoint of Donchian channel
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midpoint
                if close[i] < donchian_mid_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above midpoint
                if close[i] > donchian_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_20_1wEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0