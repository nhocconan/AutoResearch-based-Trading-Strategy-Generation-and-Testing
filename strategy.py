#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + Choppiness Regime Filter
Hypothesis: Donchian channel breakouts capture strong momentum moves. Volume confirmation ensures participation. 
Choppiness regime filter (CHOP > 61.8 = ranging, CHOP < 38.2 = trending) avoids false breakouts in sideways markets.
Works in bull (long on upper break) and bear (short on lower break). Target: 20-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for choppiness filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Choppiness Index on 1d (period=14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14)
    atr_14 = np.full_like(close_1d, np.nan)
    for i in range(14, len(tr)):
        atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    # Sum of ATR over period
    sum_atr = np.full_like(close_1d, np.nan)
    for i in range(14, len(atr_14)):
        sum_atr[i] = np.nansum(atr_14[i-13:i+1])
    
    # Max high - min low over period
    max_high_minus_min_low = np.full_like(close_1d, np.nan)
    for i in range(14, len(high_1d)):
        max_high = np.nanmax(high_1d[i-13:i+1])
        min_low = np.nanmin(low_1d[i-13:i+1])
        max_high_minus_min_low[i] = max_high - min_low
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR) / (max_high - min_low)) / log10(period)
    chop = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        if sum_atr[i] > 0 and max_high_minus_min_low[i] > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / max_high_minus_min_low[i]) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian(20) on primary timeframe (4h)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, volume MA, and chop
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Regime filter: only trade when CHOP < 50 (trending market)
        regime_filter = chop_val < 50
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high, volume confirmation, trending regime
            long_entry = (curr_close > donch_high_val) and volume_confirm and regime_filter
            # Short: price breaks below Donchian low, volume confirmation, trending regime
            short_entry = (curr_close < donch_low_val) and volume_confirm and regime_filter
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price closes below Donchian low (trailing stop)
            if curr_close < donch_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price closes above Donchian high (trailing stop)
            if curr_close > donch_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0