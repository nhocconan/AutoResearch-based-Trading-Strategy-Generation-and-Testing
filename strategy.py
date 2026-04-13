#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h volume spike and chop regime filter
    # Donchian breakout provides directional edge, volume confirms institutional participation,
    # chop regime avoids whipsaws in ranging markets. Works in both bull/bear via breakouts.
    # Target: 20-50 trades/year (80-200 total over 4 years) for low fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF indicators
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ATR(14) for chop regime
    tr_12h = np.maximum(
        high_12h[1:] - low_12h[1:],
        np.maximum(
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
    )
    tr_12h = np.concatenate([[np.nan], tr_12h])  # align with index 0
    
    atr_12h = np.full(len(df_12h), np.nan)
    for i in range(14, len(tr_12h)):
        atr_12h[i] = np.nanmean(tr_12h[i-13:i+1])
    
    # Calculate 12h Donchian(20) range for chop denominator
    donchian_high_12h = np.full(len(df_12h), np.nan)
    donchian_low_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        donchian_high_12h[i] = np.max(high_12h[i-20:i])
        donchian_low_12h[i] = np.min(low_12h[i-20:i])
    
    # Chop = ATR / (Donchian High - Donchian Low), higher = choppier
    chop_denom = donchian_high_12h - donchian_low_12h
    chop_12h = np.full(len(df_12h), np.nan)
    for i in range(len(df_12h)):
        if chop_denom[i] > 0:
            chop_12h[i] = (atr_12h[i] / chop_denom[i]) * 100
    
    # Choppiness regime: CHOP > 61.8 = range (avoid breakouts), CHOP < 38.2 = trending (favor breakouts)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Get 12h volume MA(20) for volume spike confirmation
    vol_ma_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    vol_spike_12h = volume_12h > (2.0 * vol_ma_12h)  # 2x average = significant spike
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.astype(float))
    
    # Calculate 4h Donchian(20) for breakout signals
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending regime (CHOP < 38.2) with volume spike
        trending_regime = chop_aligned[i] < 38.2
        volume_confirmation = vol_spike_aligned[i] > 0.5  # boolean as float
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Entry logic: Breakout + volume confirmation + trending regime
        long_entry = long_breakout and volume_confirmation and trending_regime
        short_entry = short_breakout and volume_confirmation and trending_regime
        
        # Exit logic: opposite breakout or chop regime shifts to extreme range
        long_exit = short_breakout or (chop_aligned[i] > 61.8)
        short_exit = long_breakout or (chop_aligned[i] > 61.8)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_breakout_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0