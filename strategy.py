#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian(20) breakout with weekly EMA20 trend filter and volume confirmation.
Long when price breaks above 20-day high with rising weekly EMA20 and volume spike.
Short when price breaks below 20-day low with falling weekly EMA20 and volume spike.
Exit when price returns to the 20-day midpoint.
Designed for low trade frequency by requiring multiple confirmations and using daily/weekly timeframes.
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
    
    # Load daily data for Donchian channels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-day high and low (using previous day's data to avoid look-ahead)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Align daily Donchian levels to 15m timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_1d, mid_20)
    
    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or np.isnan(mid_20_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above 20-day high with rising weekly EMA20 and volume spike
            if (close[i] > high_20_aligned[i] and 
                ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with falling weekly EMA20 and volume spike
            elif (close[i] < low_20_aligned[i] and 
                  ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to 20-day midpoint
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

name = "1D_Donchian_20_1wEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0