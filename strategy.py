#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + Choppiness Regime Filter
Hypothesis: Donchian channel breakouts capture strong momentum moves, especially when
confirmed by volume expansion and occurring in trending (low chop) regimes.
This strategy targets 20-40 trades per year on 4h timeframe by requiring:
1. Price breaks above/below 20-period Donchian channel
2. Volume > 2.0 x 20-period volume MA (volume confirmation)
3. Choppiness Index < 38.2 (trending regime filter)
4. Position size 0.25 to limit drawdown and fee churn
Works in both bull and bear markets: choppiness filter avoids whipsaws in ranging markets,
while volume confirmation ensures breakouts have conviction. Discrete sizing minimizes fees.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels on primary timeframe
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])  # align with price indices
    
    atr_14 = np.full(n, np.nan)
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    max_high_14 = np.full(n, np.nan)
    min_low_14 = np.full(n, np.nan)
    for i in range(14, n):
        max_high_14[i] = np.max(high[i-13:i+1])
        min_low_14[i] = np.min(low[i-13:i+1])
    
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if max_high_14[i] > min_low_14[i] and not np.isnan(atr_14[i]):
            sum_atr = np.sum(atr_14[i-13:i+1])
            range_14 = max_high_14[i] - min_low_14[i]
            chop[i] = 100 * np.log10(sum_atr / range_14) / np.log10(14)
        else:
            chop[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20), volume MA (20), and chop (14)
    start_idx = max(20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Regime filter: Choppiness Index < 38.2 (trending market)
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Look for entry signals
            # Long: Break above Donchian high AND volume confirmation AND trending regime
            long_entry = (curr_high > donch_high[i] and 
                         volume_confirm and trending_regime)
            # Short: Break below Donchian low AND volume confirmation AND trending regime
            short_entry = (curr_low < donch_low[i] and 
                          volume_confirm and trending_regime)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Price closes below Donchian low (channel breakdown)
            if curr_close < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Price closes above Donchian high (channel breakdown)
            if curr_close > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChoppinessFilter"
timeframe = "4h"
leverage = 1.0