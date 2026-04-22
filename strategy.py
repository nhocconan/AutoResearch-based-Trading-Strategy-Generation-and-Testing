#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation
# Choppiness Index > 61.8 indicates ranging market (mean revert at Donchian bands)
# Choppiness Index < 38.2 indicates trending market (breakout follow)
# In ranging markets: buy at lower band, sell at upper band
# In trending markets: buy on breakout above upper band, sell on breakout below lower band
# Volume confirmation (>1.5x 20-period average) filters false breakouts
# Designed for 4h timeframe targeting 20-30 trades/year with strong performance in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Choppiness Index calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14-period) on daily data
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (max(high) - min(low)))) / log10(n)
    # Simplified: CHOP = 100 * log10(atr_sum / (highest_high - lowest_low)) / log10(14)
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr1 = np.concatenate([[high_1d[0] - low_1d[0]], tr1])
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # For each period, calculate sum of ATR(14) over last 14 periods
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period) on 4h data
    highest_high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for all indicators
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(highest_high_4h[i]) or 
            np.isnan(lowest_low_4h[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        donchian_upper = highest_high_4h[i]
        donchian_lower = lowest_low_4h[i]
        
        if position == 0:
            # Ranging market (CHOP > 61.8): mean reversion at Donchian bands
            if chop_val > 61.8:
                # Buy near lower band, sell near upper band
                if close[i] <= donchian_lower * 1.001:  # Slight buffer for entry
                    if volume[i] > 1.5 * vol_avg_20[i]:  # Volume confirmation
                        signals[i] = 0.25
                        position = 1
                elif close[i] >= donchian_upper * 0.999:  # Slight buffer for entry
                    if volume[i] > 1.5 * vol_avg_20[i]:  # Volume confirmation
                        signals[i] = -0.25
                        position = -1
            # Trending market (CHOP < 38.2): breakout follow
            elif chop_val < 38.2:
                # Buy on breakout above upper band
                if close[i] > donchian_upper:
                    if volume[i] > 1.5 * vol_avg_20[i]:  # Volume confirmation
                        signals[i] = 0.25
                        position = 1
                # Sell on breakout below lower band
                elif close[i] < donchian_lower:
                    if volume[i] > 1.5 * vol_avg_20[i]:  # Volume confirmation
                        signals[i] = -0.25
                        position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price reaches opposite band or regime change to trending down
                if close[i] >= donchian_upper * 0.999 or chop_val < 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price reaches opposite band or regime change to trending up
                if close[i] <= donchian_lower * 1.001 or chop_val < 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_Donchian20_VolumeConfirm"
timeframe = "4h"
leverage = 1.0