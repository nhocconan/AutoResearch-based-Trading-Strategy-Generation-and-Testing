#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation.
# Uses Choppiness Index (14) to detect ranging markets (CHOP > 61.8) and take mean-reversion trades at Donchian bands.
# In trending markets (CHOP < 38.2), breakouts are traded in direction of trend.
# Volume spike confirms momentum. Designed to reduce false signals and whipsaws in both bull and bear markets.
# Target: 20-40 trades/year per symbol (80-160 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on same timeframe
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14)
    atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (max_high14 - min_low14)) / np.log10(14)
    
    # Volume spike (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(chop[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean-reversion in ranging market (CHOP > 61.8)
            if chop[i] > 61.8:
                if close[i] <= low_20[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= high_20[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # Breakout in trending market (CHOP < 38.2)
            elif chop[i] < 38.2:
                if close[i] > high_20[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                if close[i] >= high_20[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] <= low_20[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_Regime_Donchian20_Breakout_MeanRev_Volume"
timeframe = "4h"
leverage = 1.0