#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 4h Donchian breakout and volume spike.
# Uses Choppiness Index (14) to detect ranging vs trending regimes:
#   CHOP > 61.8 = ranging (mean reversion at Donchian bands)
#   CHOP < 38.2 = trending (breakout continuation)
# Combines with Donchian(20) breakouts and volume confirmation for high-probability entries.
# Designed to work in bull (trending breakouts) and bear (mean reversion in ranges).
# Target: 20-40 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for Choppiness Index
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index (14)
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    
    # Calculate Donchian channels (20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need Donchian20 and chop
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Price relative to Donchian channels
        price_at_upper = close[i] >= donch_high[i]
        price_at_lower = close[i] <= donch_low[i]
        
        # Regime filters
        chop_ranging = chop[i] > 61.8  # Ranging market
        chop_trending = chop[i] < 38.2  # Trending market
        
        if position == 0:
            # Long in trending market: break above Donchian high with volume
            if (chop_trending and price_at_upper and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short in trending market: break below Donchian low with volume
            elif (chop_trending and price_at_lower and volume_filter):
                signals[i] = -0.25
                position = -1
            # Long in ranging market: mean reversion from Donchian low
            elif (chop_ranging and price_at_lower and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short in ranging market: mean reversion from Donchian high
            elif (chop_ranging and price_at_upper and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian low OR chop shifts to strong trending
            if (close[i] <= donch_low[i]) or (chop[i] < 25.0):  # Strong trend, consider trailing
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian high OR chop shifts to strong trending
            if (close[i] >= donch_high[i]) or (chop[i] < 25.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Chop_DonchianBreakout_Volume"
timeframe = "4h"
leverage = 1.0