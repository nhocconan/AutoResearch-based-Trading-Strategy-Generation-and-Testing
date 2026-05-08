#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 12h Donchian breakout with volume confirmation
# Uses Choppiness Index (14) to filter trending vs ranging markets, Donchian(20) breakout for entry,
# and volume spike (>2x 20-period average) for confirmation. Designed to capture trends in both
# bull and bear markets with strict entry conditions to limit trades to 12-37/year.

name = "12h_ChopFilter_Donchian20_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Choppiness Index (14)
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[np.nan], tr])
        atr_sum = np.full(n, np.nan)
        if n >= 14:
            atr_sum[13] = np.nansum(tr[1:15])
            for i in range(14, n):
                atr_sum[i] = atr_sum[i-1] - tr[i-13] + tr[i]
            atr[13:] = atr_sum[13:] / 14
    
    max_high = np.full(n, np.nan)
    min_low = np.full(n, np.nan)
    if n >= 14:
        for i in range(14, n):
            max_high[i] = np.max(high[i-13:i+1])
            min_low[i] = np.min(low[i-13:i+1])
    
    chop = np.full(n, np.nan)
    if n >= 14:
        for i in range(14, n):
            if not np.isnan(atr[i]) and not np.isnan(max_high[i]) and not np.isnan(min_low[i]):
                if max_high[i] > min_low[i]:
                    chop[i] = 100 * np.log10(atr[i] / (max_high[i] - min_low[i])) / np.log10(14)
    
    # Calculate 12h Donchian(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 12h volume average for volume spike
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(chop[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop regime: < 38.2 = trending, > 61.8 = ranging
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        if position == 0:
            # Look for entry: Donchian breakout in trending market with volume confirmation
            if is_trending:
                if close[i] > donchian_high[i] and volume[i] > 2.0 * vol_avg_20[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i] and volume[i] > 2.0 * vol_avg_20[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: trend ends or price retrace to Donchian midpoint
            if not is_trending or close[i] < (donchian_high[i] + donchian_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend ends or price retrace to Donchian midpoint
            if not is_trending or close[i] > (donchian_high[i] + donchian_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals