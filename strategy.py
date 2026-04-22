#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout
    # Choppiness > 61.8 indicates ranging market (mean reversion opportunity)
    # Choppiness < 38.2 indicates trending market (breakout continuation)
    # Donchian breakout with volume confirmation provides entry in correct regime
    # Target: 20-40 trades/year with regime adaptation
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Choppiness Index calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(14) for Choppiness Index
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index: 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_raw = np.where((max_high - min_low) > 0, chop_raw, 50)  # avoid division by zero
    
    chop_4h = chop_raw
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(chop_4h_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop = chop_4h_aligned[i]
        
        if position == 0:
            # Ranging market (Chop > 61.8): mean reversion at Donchian channels
            if chop > 61.8:
                # Long: bounce from lower Donchian band with volume
                if close[i] <= donch_low[i] * 1.001 and vol_spike[i]:  # near or at lower band
                    signals[i] = 0.25
                    position = 1
                # Short: bounce from upper Donchian band with volume
                elif close[i] >= donch_high[i] * 0.999 and vol_spike[i]:  # near or at upper band
                    signals[i] = -0.25
                    position = -1
            # Trending market (Chop < 38.2): breakout continuation
            elif chop < 38.2:
                # Long: break above upper Donchian band with volume
                if close[i] > donch_high[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower Donchian band with volume
                elif close[i] < donch_low[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: return to middle of Donchian channel or reverse signal
                donch_mid = (donch_high[i] + donch_low[i]) / 2
                if close[i] < donch_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: return to middle of Donchian channel or reverse signal
                donch_mid = (donch_high[i] + donch_low[i]) / 2
                if close[i] > donch_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_Donchian_Breakout_Volume_v1"
timeframe = "4h"
leverage = 1.0