#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian channel breakout with 1-week EMA200 trend filter and volume confirmation.
Long when price breaks above Donchian(20) high, price > EMA200(1w), and volume > 1.5x average.
Short when price breaks below Donchian(20) low, price < EMA200(1w), and volume > 1.5x average.
Exit when price returns to Donchian middle or EMA200(1w) trend fails.
Designed for low trade frequency (~10-25/year) to capture major trends while minimizing whipsaws.
Works in both bull and bear markets by requiring strong trend filter (EMA200).
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
    
    # Load 1-week data for EMA200 - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1-week EMA200
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate 1-day Donchian channels (20-period)
    # Upper band: highest high of last 20 days
    # Lower band: lowest low of last 20 days
    # Middle band: average of upper and lower
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200_val = ema200_1w_aligned[i]
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high, above EMA200, volume confirmation
            if (high_val > donchian_high[i] and 
                close_val > ema200_val and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, below EMA200, volume confirmation
            elif (low_val < donchian_low[i] and 
                  close_val < ema200_val and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to Donchian middle OR price below EMA200
                if (close_val <= donchian_mid[i]) or (close_val < ema200_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to Donchian middle OR price above EMA200
                if (close_val >= donchian_mid[i]) or (close_val > ema200_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_EMA200_1w_Volume"
timeframe = "1d"
leverage = 1.0