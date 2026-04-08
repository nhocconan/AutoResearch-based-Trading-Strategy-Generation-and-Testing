#!/usr/bin/env python3
# 1d_1w_donchian_volume_chop_v1
# Hypothesis: Daily Donchian breakout with weekly chop regime filter and volume confirmation.
# Long: price breaks above 20-day Donchian high AND weekly chop < 61.8 (trending) AND volume > 1.5x average
# Short: price breaks below 20-day Donchian low AND weekly chop < 61.8 AND volume > 1.5x average
# Exit: opposite Donchian breakout or chop > 61.8 (range) triggers flat.
# Designed to capture strong trends in both bull and bear markets while avoiding whipsaws in ranging markets.
# Weekly chop filter ensures we only trade when higher timeframe is trending, reducing false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Weekly chop regime (choppiness index)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True range
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], 
                       np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), 
                                  np.abs(low_1w[1:] - close_1w[:-1])))
    tr_1w = np.concatenate([[np.nan], tr_1w])  # align with index 0
    
    # ATR(14) weekly
    atr_1w = np.full(len(tr_1w), np.nan)
    for i in range(14, len(tr_1w)):
        atr_1w[i] = np.nanmean(tr_1w[i-13:i+1])
    
    # Highest high and lowest low over 14 weeks
    hh_1w = np.full(len(high_1w), np.nan)
    ll_1w = np.full(len(low_1w), np.nan)
    for i in range(14, len(high_1w)):
        hh_1w[i] = np.max(high_1w[i-14:i+1])
        ll_1w[i] = np.min(low_1w[i-14:i+1])
    
    # Chop = 100 * log10(sum(TR)/ (HH-LL)) / log10(14)
    chop_1w = np.full(len(high_1w), np.nan)
    for i in range(14, len(high_1w)):
        if hh_1w[i] > ll_1w[i] and not np.isnan(atr_1w[i]) and atr_1w[i] > 0:
            sum_tr = np.nansum(tr_1w[i-13:i+1])
            chop_1w[i] = 100 * np.log10(sum_tr / (hh_1w[i] - ll_1w[i])) / np.log10(14)
    
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    volume_ratio = volume / avg_volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(chop_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol_confirmed = volume_ratio[i] > 1.5
        trending_regime = chop_1w_aligned[i] < 61.8  # chop < 61.8 = trending
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR chop becomes ranging
            if price < lowest_low[i] or chop_1w_aligned[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR chop becomes ranging
            if price > highest_high[i] or chop_1w_aligned[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high + volume + trending regime
            if price > highest_high[i] and vol_confirmed and trending_regime:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low + volume + trending regime
            elif price < lowest_low[i] and vol_confirmed and trending_regime:
                position = -1
                signals[i] = -0.25
    
    return signals