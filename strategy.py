#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter combined with 1d Donchian breakout and volume confirmation.
# In high volatility regimes (CHOP < 38.2), trade breakouts of 1d Donchian channels (20-period).
# In low volatility regimes (CHOP > 61.8), avoid trading to prevent whipsaws.
# Volume confirmation ensures breakout legitimacy.
# Designed for 4h timeframe with controlled trade frequency (target: 20-50/year) to avoid fee drag.
# Works in both bull and bear markets by adapting to volatility regimes.

name = "4h_Chop_DonchianBreakout_1dVolume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14-period) for regime detection
    atr14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    atr14[0] = np.mean(np.maximum(np.maximum(high[:1] - low[:1], np.abs(high[:1] - np.roll(close[:1], 1))), np.abs(low[:1] - np.roll(close[:1], 1)))) if n > 0 else 0
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / (max_high - min_low)) / np.log10(14)
    chop = np.where((max_high - min_low) != 0, chop, 50)  # Avoid division by zero
    
    # 1d Donchian channels (20-period) for breakout signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(chop[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Low volatility regime (CHOP > 61.8), price breaks above 1d Donchian high, volume confirmation
            long_cond = (chop[i] > 61.8) and (close[i] > donchian_high_aligned[i]) and volume_filter[i]
            # Short conditions: Low volatility regime (CHOP > 61.8), price breaks below 1d Donchian low, volume confirmation
            short_cond = (chop[i] > 61.8) and (close[i] < donchian_low_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: High volatility regime (CHOP < 38.2) OR price breaks below Donchian low
            if (chop[i] < 38.2) or (close[i] < donchian_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: High volatility regime (CHOP < 38.2) OR price breaks above Donchian high
            if (chop[i] < 38.2) or (close[i] > donchian_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals