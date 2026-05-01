#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d Williams %R(14) extreme reversal filter
# Uses 1d Williams %R to identify overbought/oversold conditions on daily timeframe
# Enter long when price breaks above Donchian upper band AND 1d %R < -80 (oversold)
# Enter short when price breaks below Donchian lower band AND 1d %R > -20 (overbought)
# Volume confirmation (1.5x 20-period MA) ensures participation
# Works in bull/bear markets by fading extremes during breakouts
# Targets 50-150 total trades over 4 years to minimize fee drag on 6h timeframe

name = "6h_Donchian20_1dWilliamsR_Extreme_Reversal_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, ((highest_high - close_1d) / denominator) * -100, -50)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 6h Donchian channels (20-period)
    highest_high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_6h
    donchian_lower = lowest_low_6h
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20)  # Need Williams %R and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > donchian_upper[i]  # Price breaks above Donchian upper
        breakout_short = close[i] < donchian_lower[i]  # Price breaks below Donchian lower
        
        # Williams %R extremes
        williams_oversold = williams_r_aligned[i] < -80  # Oversold condition
        williams_overbought = williams_r_aligned[i] > -20  # Overbought condition
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian upper with oversold 1d %R and volume spike
            if breakout_long and williams_oversold and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian lower with overbought 1d %R and volume spike
            elif breakout_short and williams_overbought and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below Donchian middle or Williams %R > -50 (momentum fade)
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] < donchian_middle or williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above Donchian middle or Williams %R < -50 (momentum fade)
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] > donchian_middle or williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals