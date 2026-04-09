#!/usr/bin/env python3
# 1d_donchian_breakout_volume_chop_regime_v1
# Hypothesis: 1d strategy using Donchian(20) breakouts for entry, volume confirmation (>1.5x 20-bar average volume), and choppiness regime filter (CHOP(14) > 61.8 = range → mean reversion, CHOP < 38.2 = trending → trend follow). Enters long on upper band breakout with volume confirmation in trending regime; enters short on lower band breakout with volume confirmation in trending regime. Uses weekly timeframe only for HTF alignment safety. Target: 7-25 trades/year (30-100 total over 4 years). Donchian provides clear structure, volume filters weak breakouts, chop regime avoids whipsaws in sideways markets. Works in bull/bear by adapting to market regime.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_chop_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period = 20 days)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Multi-timeframe: 1w HTF alignment safety (not used for signals, just for alignment)
    df_1w = get_htf_data(prices, '1w')
    
    # Choppiness Index (14-period)
    def calculate_chop(high, low, close, window=14):
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=window, min_periods=1).sum()
        max_high = pd.Series(high).rolling(window=window, min_periods=1).max()
        min_low = pd.Series(low).rolling(window=window, min_periods=1).min()
        chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(window)
        # Handle division by zero and edge cases
        chop = np.where((max_high - min_low) == 0, 50, chop)
        return chop.values
    
    chop = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filters
        is_trending = chop[i] < 38.2  # Trending regime
        is_ranging = chop[i] > 61.8   # Ranging regime
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band OR regime shifts to ranging
            if close[i] < low_min[i] or is_ranging:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band OR regime shifts to ranging
            if close[i] > high_max[i] or is_ranging:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation in trending regime
            if is_trending and volume_confirmed:
                if close[i] > high_max[i]:  # Breakout above upper band
                    position = 1
                    signals[i] = 0.25
                elif close[i] < low_min[i]:  # Breakdown below lower band
                    position = -1
                    signals[i] = -0.25
    
    return signals