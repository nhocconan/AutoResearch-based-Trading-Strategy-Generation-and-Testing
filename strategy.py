#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v6
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-bar average volume) and
# chop regime filter (Choppiness Index > 61.8 = range, < 38.2 = trend). Only take breakouts
# aligned with 1d EMA(50) trend. Long: price breaks above upper band + volume + chop < 38.2 + price > 1d EMA50.
# Short: price breaks below lower band + volume + chop < 38.2 + price < 1d EMA50.
# Exit: opposite Donchian breakout or chop > 61.8 (range regime). Uses discrete positions ±0.25.
# Designed for low trade frequency (<40/year) and works in bull/bear via 1d EMA trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v6"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period)
    atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (highest_high_14 - lowest_low_14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if np.isnan(ema_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR chop > 61.8 (range regime)
            if close[i] < lowest_low[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR chop > 61.8 (range regime)
            if close[i] > highest_high[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed and chop[i] < 38.2:  # Only in trending regime (chop < 38.2)
                # Long breakout: price above upper Donchian + above 1d EMA50
                if close[i] > highest_high[i] and close[i] > ema_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price below lower Donchian + below 1d EMA50
                elif close[i] < lowest_low[i] and close[i] < ema_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals