#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and Choppiness index regime filter.
# Long when price breaks above Donchian upper band with volume spike and trending regime (CHOP < 38.2).
# Short when price breaks below Donchian lower band with volume spike and trending regime (CHOP < 38.2).
# Uses volume spike (>1.5x 20-period average) to confirm breakout strength.
# Uses Choppiness index (CHOP < 38.2) to identify trending markets and avoid range-bound false breakouts.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years).
name = "4h_Donchian20_Volume_ChopFilter"
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
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * true_range)) / log10(n)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # First TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # True range for denominator
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    chop = 100 * np.log10(tr_sum / (14 * atr1)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need Donchian and CHOP data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        chop_val = chop[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Trending regime filter (CHOP < 38.2 indicates trending market)
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Enter long: price breaks above Donchian upper band + volume + trend
            if price > upper and volume_confirmed and trending_regime:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower band + volume + trend
            elif price < lower and volume_confirmed and trending_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below Donchian lower band
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above Donchian upper band
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals