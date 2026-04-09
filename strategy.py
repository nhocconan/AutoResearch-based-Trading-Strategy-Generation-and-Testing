#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v4
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Works in bull/bear: Donchian(20) captures breakouts; volume > 1.5x 20-ma confirms institutional participation;
# Choppiness Index(14) < 38.2 filters for trending markets only, reducing whipsaws in ranging conditions.
# Target: 20-50 trades/year, discrete sizing 0.25 for cost control.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v4"
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
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14) - values < 38.2 indicate trending market
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (HH(14) - LL(14))))
    # Simplified: CHOP = 100 * LOG10( SUM(TR(14)) / (LOG10(14) * (HH(14) - LL(14))) )
    # We'll use a common approximation: CHOP = 100 * LOG10( SUM(TR(14)) / (LOG10(14) * (HH(14) - LL(14))) )
    # For efficiency: TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # first bar TR
    
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = np.log10(14) * (hh14 - ll14)
    chop = np.where(denominator != 0, 
                    100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / denominator), 
                    100)  # Set to 100 (ranging) when denominator is zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR chop becomes too high (ranging)
            if close[i] < lowest_low[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR chop becomes too high (ranging)
            if close[i] > highest_high[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation AND trending market (chop < 38.2)
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            trending_market = chop[i] < 38.2
            
            if volume_confirmed and trending_market:
                # Long: price breaks above Donchian high
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals