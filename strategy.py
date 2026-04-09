#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and chop regime filter
# Donchian breakouts provide clear structure for trend continuation in both bull/bear markets
# 12h volume confirmation (>1.5x 20-period average) filters false breakouts
# Chop regime filter (CHOP(14) > 61.8) avoids trading in strong trends where breakouts fail
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# Works in bull/bear: breakouts capture trends, volume confirms validity, chop filter avoids whipsaws
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "4h_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Pre-compute indicators
    # Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume confirmation
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Choppiness Index (14-period) for regime filter
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.append([np.nan], close[:-1]))), np.abs(low - np.append([np.nan], close[:-1])))).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr_14 * 14) / (np.log(highest_high_14 - lowest_low_14))) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume
        volume_confirmed = volume_12h[i] > 1.5 * vol_ma_20_12h_aligned[i]
        
        # Chop regime filter: only trade when market is ranging/choppy (CHOP > 61.8)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit on Donchian lower band retracement
            if close[i] < lowest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on Donchian upper band retracement
            if close[i] > highest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volume and chop confirmation
            # Long on Donchian upper breakout, Short on Donchian lower breakout
            if volume_confirmed and chop_filter:
                if close[i] > highest_20[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lowest_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals