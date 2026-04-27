#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter with 1w Donchian breakout.
# In trending markets (CHOP < 38.2), trade breakouts of weekly Donchian channels.
# In ranging markets (CHOP > 61.8), fade the extremes with mean reversion.
# Uses volume confirmation to avoid false breakouts.
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 10-25 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) for daily
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    atr_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.nanmean(tr[1:i+1])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate Choppiness Index
    sum_atr = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        sum_atr[i] = np.nansum(atr_14[i-13:i+1])
    
    highest_high = np.full(len(df_1d), np.nan)
    lowest_low = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        highest_high[i] = np.nanmax(high_1d[i-13:i+1])
        lowest_low[i] = np.nanmin(low_1d[i-13:i+1])
    
    chop = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if highest_high[i] > lowest_low[i] and sum_atr[i] > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian(20) channels
    donch_high = np.full(len(df_1w), np.nan)
    donch_low = np.full(len(df_1w), np.nan)
    for i in range(19, len(df_1w)):
        donch_high[i] = np.nanmax(high_1w[i-19:i+1])
        donch_low[i] = np.nanmin(low_1w[i-19:i+1])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Volume confirmation: current volume > 1.3 * 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.nanmean(volume[i-20:i])
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        
        # Determine market regime
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        is_neutral = not (is_trending or is_ranging)
        
        if position == 0:
            if is_trending:
                # Trending market: trade Donchian breakouts
                if (high[i] > donch_high_aligned[i] and 
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                elif (low[i] < donch_low_aligned[i] and 
                      volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging market: fade extremes
                if (low[i] <= donch_low_aligned[i] and 
                    volume_confirm[i]):
                    signals[i] = 0.20  # long at lower band
                    position = 1
                elif (high[i] >= donch_high_aligned[i] and 
                      volume_confirm[i]):
                    signals[i] = -0.20  # short at upper band
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Neutral: no action
                signals[i] = 0.0
        elif position == 1:
            # Long exit conditions
            if is_trending:
                # Exit trend trade on opposite Donchian touch
                if low[i] <= donch_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif is_ranging:
                # Exit range trade at midpoint or opposite extreme
                midpoint = (donch_high_aligned[i] + donch_low_aligned[i]) / 2
                if close[i] >= midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:
                signals[i] = 0.20 if position == 1 else -0.20
        elif position == -1:
            # Short exit conditions
            if is_trending:
                # Exit trend trade on opposite Donchian touch
                if high[i] >= donch_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif is_ranging:
                # Exit range trade at midpoint or opposite extreme
                midpoint = (donch_high_aligned[i] + donch_low_aligned[i]) / 2
                if close[i] <= midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1d_ChopRegime_DonchianBreakout_1wVolume_v1"
timeframe = "1d"
leverage = 1.0