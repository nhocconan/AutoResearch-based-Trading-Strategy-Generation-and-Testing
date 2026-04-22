#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Choppiness Index regime filter + 1d Donchian(20) breakout + volume spike
    # Choppiness Index > 61.8 = ranging market (mean revert at Donchian bands)
    # Choppiness Index < 38.2 = trending market (follow Donchian breakout)
    # Donchian breakout with volume confirmation provides institutional-grade entries
    # Works in bull/bear: adapts to regime - mean revert in range, follow trend in trend
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channels and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (14-period)
    atr_1d = np.maximum(high_1d - low_1d,
                        np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                   np.abs(low_1d - np.roll(close_1d, 1))))
    atr_1d[0] = high_1d[0] - low_1d[0]
    
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high - lowest_low
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(atr_sum / range_14) / np.log10(14)
    
    # Align Donchian and Choppiness to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # In ranging market (CHOP > 61.8): mean revert at Donchian bands
            if chop_aligned[i] > 61.8:
                # Long: bounce from lower band with volume spike
                if close[i] <= donchian_low_aligned[i] * 1.001 and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: bounce from upper band with volume spike
                elif close[i] >= donchian_high_aligned[i] * 0.999 and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # In trending market (CHOP < 38.2): follow Donchian breakout
            elif chop_aligned[i] < 38.2:
                # Long: break above upper band with volume spike
                if close[i] > donchian_high_aligned[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower band with volume spike
                elif close[i] < donchian_low_aligned[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price reaches opposite band or chop shifts to extreme trending
                if close[i] >= donchian_high_aligned[i] or chop_aligned[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price reaches opposite band or chop shifts to extreme trending
                if close[i] <= donchian_low_aligned[i] or chop_aligned[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Chop_Regime_Donchian20_Breakout_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0