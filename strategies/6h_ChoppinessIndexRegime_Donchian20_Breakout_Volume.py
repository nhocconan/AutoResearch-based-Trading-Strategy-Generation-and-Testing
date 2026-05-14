#!/usr/bin/env python3
# 6h_ChoppinessIndexRegime_Donchian20_Breakout_Volume
# Hypothesis: Combines Choppiness Index (CI) regime filter with Donchian channel breakout on 6h timeframe.
# CI(14) > 61.8 indicates ranging market (mean reversion) - fade Donchian breakouts.
# CI(14) < 38.2 indicates trending market - trade Donchian breakouts in direction of trend.
# Volume confirmation required (>1.5x average volume).
# Designed for 15-35 trades/year to avoid overtrading and work in both bull and bear markets.

name = "6h_ChoppinessIndexRegime_Donchian20_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 60-period ATR for Choppiness Index
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_60 = np.full(n, np.nan)
    for i in range(60, n):
        atr_60[i] = np.nanmean(tr[i-59:i+1])
    
    # Calculate Choppiness Index (14-period)
    cp = np.full(n, np.nan)
    for i in range(74, n):  # 60 + 14 - 1
        atr_sum = np.nansum(atr_60[i-13:i+1])
        highest_high = np.nanmax(high[i-13:i+1])
        lowest_low = np.nanmin(low[i-13:i+1])
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            cp[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.nanmax(high[i-20:i])
        donchian_low[i] = np.nanmin(low[i-20:i])
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 75  # Ensure sufficient warmup for Choppiness Index
    
    for i in range(start_idx, n):
        if np.isnan(cp[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trending market regime (CI < 38.2) - trade breakouts
            if cp[i] < 38.2:
                # Long: Breakout above Donchian high with volume confirmation
                if close[i] > donchian_high[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Breakout below Donchian low with volume confirmation
                elif close[i] < donchian_low[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market regime (CI > 61.8) - fade breakouts
            elif cp[i] > 61.8:
                # Long: Fade breakdown below Donchian low with volume confirmation
                if close[i] < donchian_low[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Fade breakout above Donchian high with volume confirmation
                elif close[i] > donchian_high[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price crosses below Donchian low (trending) or above Donchian high (ranging)
            if (cp[i] < 38.2 and close[i] < donchian_low[i]) or (cp[i] > 61.8 and close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above Donchian high (trending) or below Donchian low (ranging)
            if (cp[i] < 38.2 and close[i] > donchian_high[i]) or (cp[i] > 61.8 and close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals