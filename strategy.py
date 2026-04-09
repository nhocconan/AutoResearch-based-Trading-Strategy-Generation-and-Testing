#!/usr/bin/env python3
# 4h_donchian_20_volume_chop_regime_v1
# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high with volume > 1.5x 20-period average and choppy market (CHOP > 61.8).
# Short when price breaks below Donchian(20) low with volume confirmation and choppy market.
# Uses discrete position sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_volume_chop_regime_v1"
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
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (CHOP) - 14 period
    # CHOP = 100 * log10(sum(ATR(1) over 14 periods) / log10(highest_high - lowest_low over 14 periods)) / log10(14)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # First TR is just high-low
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR chop regime ends (trending market)
            if close[i] < lowest_low_20[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR chop regime ends (trending market)
            if close[i] > highest_high_20[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and choppy market (CHOP > 61.8 = ranging)
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            choppy_market = chop[i] > 61.8
            
            if volume_confirmed and choppy_market:
                # Long: price breaks above Donchian high
                if close[i] > highest_high_20[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif close[i] < lowest_low_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals