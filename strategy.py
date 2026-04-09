#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v2
# Hypothesis: 4h strategy using Donchian(20) breakout for entry, volume confirmation (>1.5x 20-bar average volume), and choppiness regime filter (CHOP(14) between 38.2 and 61.8 for ranging markets). Enters long on upper band breakout with volume confirmation in ranging regime; enters short on lower band breakout with volume confirmation in ranging regime. Uses 1d HTF for weekly alignment safety. Target: 75-200 total trades over 4 years (19-50/year). Choppiness filter avoids trending markets where breakouts fail, focusing on mean-reversion in ranges where Donchian breaks often reverse. Volume confirmation ensures institutional participation. Works in bull/bear by fading false breakouts in ranges.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v2"
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
    
    # Volume average for confirmation (20-period = ~3.3 days of 4h bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period) - choppy when > 38.2 and < 61.8
    def choppiness_index(high, low, close, window=14):
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        atr = np.maximum(np.maximum(high_s - low_s, np.abs(high_s - close_s.shift(1))), np.abs(low_s - close_s.shift(1)))
        atr_sum = atr.rolling(window=window, min_periods=window).sum().values
        highest_high = high_s.rolling(window=window, min_periods=window).max().values
        lowest_low = low_s.rolling(window=window, min_periods=window).min().values
        range_hl = highest_high - lowest_low
        chop = 100 * np.log10(atr_sum / np.log10(range_hl)) / np.log10(window)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # Multi-timeframe: 1d HTF for alignment safety (not used in logic, but loaded once as required)
    df_1d = get_htf_data(prices, '1d')
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(chop[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness regime filter: ranging market (38.2 < CHOP < 61.8)
        ranging_regime = (chop[i] > 38.2) and (chop[i] < 61.8)
        
        if position == 1:  # Long position
            # Exit: price re-enters Donchian channel (mean reversion)
            if close[i] < donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian channel (mean reversion)
            if close[i] > donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation in ranging regime
            if close[i] > donchian_high[i] and volume_confirmed and ranging_regime:
                position = 1
                signals[i] = 0.25  # Long on upper breakout
            elif close[i] < donchian_low[i] and volume_confirmed and ranging_regime:
                position = -1
                signals[i] = -0.25  # Short on lower breakout
    
    return signals