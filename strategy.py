#!/usr/bin/env python3
"""
4h_volume_regime_donchian_v1
Hypothesis: Donchian breakout with volume confirmation and chop regime filter.
- Breakout: Price > Donchian(20) high or < Donchian(20) low
- Volume: Current volume > 1.5x 20-period average
- Regime: Chop(14) < 61.8 (trending market) to avoid false breakouts in chop
- Position: 0.25 long/short
- Target: 20-50 trades/year to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volume_regime_donchian_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Chop index (14-period) - range: 0-100
    # Chop = 100 * log10(sum(atr14) / (max(high14) - min(low14))) / log10(14)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range14 = highest_high14 - lowest_low14
    range14 = np.where(range14 == 0, 1e-10, range14)
    
    chop = 100 * np.log10(np.sum(atr14) / range14) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)  # fill NaN with neutral
    
    # Trending regime: Chop < 61.8
    trending_filter = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if filters not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trending_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR chop > 61.8 (ranging)
            if close[i] < low_roll[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR chop > 61.8 (ranging)
            if close[i] > high_roll[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high + volume + trending
            if (close[i] > high_roll[i]) and volume_filter[i] and trending_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + volume + trending
            elif (close[i] < low_roll[i]) and volume_filter[i] and trending_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals