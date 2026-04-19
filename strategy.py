#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1d Williams %R extreme + volume confirmation
# Works in both bull and bear markets: 
# - In trending markets (CHOP < 38.2): Williams %R > -20 signals momentum continuation
# - In ranging markets (CHOP > 61.8): Williams %R < -80 or > -20 signals mean reversion at extremes
# Uses tight entry conditions to limit trades (target: 15-30/year) and avoid fee drag
# Choppiness Index avoids whipsaw in sideways markets, Williams %R captures momentum extremes
name = "12h_ChopWilliams_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R (multi-timeframe analysis - ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 12h Choppiness Index (14-period)
    tr12 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr12[0] = high[0] - low[0]
    atr12 = pd.Series(tr12).rolling(window=14, min_periods=14).mean().values
    highest_high_14_12h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14_12h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr12.sum() / (highest_high_14_12h - lowest_low_14_12h)) / np.log10(14)
    # Handle division by zero and invalid cases
    chop = np.where((highest_high_14_12h - lowest_low_14_12h) == 0, 50, chop)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # 12h ATR for position sizing and stops
    atr_12h = pd.Series(tr12).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(williams_r_aligned[i]) or \
           np.isnan(chop[i]) or np.isnan(atr_12h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_12h[i]
        chop_value = chop[i]
        williams_r_value = williams_r_aligned[i]
        
        # Volume filter: current volume > 1.3x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.3 * avg_volume
        
        if position == 0:
            # Determine market regime based on Choppiness Index
            if chop_value < 38.2:  # Trending market
                # Long: Williams %R above -20 (strong momentum) + volume
                if williams_r_value > -20 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R below -80 (weak momentum) + volume
                elif williams_r_value < -80 and volume_filter:
                    signals[i] = -0.25
                    position = -1
            elif chop_value > 61.8:  # Ranging market
                # Long: Williams %R below -80 (oversold) + volume
                if williams_r_value < -80 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R above -20 (overbought) + volume
                elif williams_r_value > -20 and volume_filter:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions
            # 1. Regime change: chop moves to extreme range (61.8) - avoid whipsaw
            # 2. Williams %R reverses: crosses back through -50 (middle)
            # 3. ATR-based stop: 2.5x ATR from entry
            if (chop_value > 61.8 or 
                williams_r_value < -50 or 
                close[i] < close[i-1] - 2.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            # 1. Regime change: chop moves to extreme range (61.8) - avoid whipsaw
            # 2. Williams %R reverses: crosses back through -50 (middle)
            # 3. ATR-based stop: 2.5x ATR from entry
            if (chop_value > 61.8 or 
                williams_r_value > -50 or 
                close[i] > close[i-1] + 2.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals