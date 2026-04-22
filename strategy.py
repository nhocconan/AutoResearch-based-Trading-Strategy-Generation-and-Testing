#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian Breakout with Weekly Trend Filter and Volume Confirmation.
Long when price breaks above 20-day high during weekly uptrend with volume spike.
Short when price breaks below 20-day low during weekly downtrend with volume spike.
Exit when price returns to Donchian midpoint or weekly trend reverses.
Designed for low trade frequency (10-20 trades/year) with strong trend alignment and volume confirmation.
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
    
    # Donchian channel (20-day high/low) - using daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period high/low for Donchian channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Weekly trend filter: 21-period EMA on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + volume spike
            if close[i] > donchian_high_aligned[i] and ema21_1w_aligned[i] > ema21_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + weekly downtrend + volume spike
            elif close[i] < donchian_low_aligned[i] and ema21_1w_aligned[i] < ema21_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Donchian midpoint or weekly trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below midpoint or weekly trend turns down
                if close[i] < donchian_mid_aligned[i] or ema21_1w_aligned[i] < ema21_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above midpoint or weekly trend turns up
                if close[i] > donchian_mid_aligned[i] or ema21_1w_aligned[i] > ema21_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0