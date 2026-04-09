#!/usr/bin/env python3
# 4h_donchian_volume_chop_regime_v3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and
# choppiness regime filter (CHOP(14) between 38.2 and 61.8 for ranging markets). Long on upper
# band breakout, short on lower band breakout. Uses discrete sizing (0.0, ±0.25) to minimize
# fee churn. Target: 20-50 trades/year. Works in both bull (breakouts) and bear (mean reversion
# in chop regime) markets by aligning with structure and filtering false signals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_regime_v3"
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
    
    # Donchian(20) on 4h
    highest = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14) - ranging market filter
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(atr14/(hh14-ll14)) / log10(14)
    # Avoid division by zero
    range_14 = hh14 - ll14
    chop = np.zeros_like(atr14)
    mask = range_14 > 0
    chop[mask] = 100 * np.log10(atr14[mask] / range_14[mask]) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness regime filter: only trade in ranging markets (38.2 <= CHOP <= 61.8)
        chop_filter = (chop[i] >= 38.2) and (chop[i] <= 61.8)
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian lower band OR chop exits range (trending)
            if close[i] < lowest[i] or not chop_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper band OR chop exits range (trending)
            if close[i] > highest[i] or not chop_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_filter:
                # Long entry: price breaks above upper Donchian band
                if close[i] > highest[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below lower Donchian band
                elif close[i] < lowest[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals