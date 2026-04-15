#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volatility filter and volume confirmation
# Uses 20-period Donchian channels for breakout signals, filtered by 12h ATR volatility
# (only trade when volatility is expanding) and volume spikes. Works in both bull and bear
# markets by trading breakouts in the direction of the trend. Target: 80-150 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channels (20-period high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load 12h data for ATR volatility filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 20-period Donchian channels on daily data
    # Upper band = highest high of last 20 days
    # Lower band = lowest low of last 20 days
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate ATR (14-period) on 12h for volatility filter
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_aligned[i])):
            continue
        
        # Calculate 20-period ATR average for volatility comparison
        atr_avg = np.mean(atr_aligned[max(0, i-20):i+1])
        volatility_expanding = atr_aligned[i] > atr_avg  # Trade when volatility is above average
        
        # Volume confirmation: current volume > 1.5x average of last 20 periods
        vol_ma = np.mean(volume[max(0, i-20):i+1])
        volume_spike = volume[i] > 1.5 * vol_ma
        
        # Long entry: price breaks above Donchian high + volatility expanding + volume spike
        if (close[i] > donchian_high_aligned[i] and
            volatility_expanding and
            volume_spike and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volatility expanding + volume spike
        elif (close[i] < donchian_low_aligned[i] and
              volatility_expanding and
              volume_spike and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or volatility contraction
        elif position == 1 and close[i] < donchian_low_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donchian_high_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volatility_Volume"
timeframe = "4h"
leverage = 1.0