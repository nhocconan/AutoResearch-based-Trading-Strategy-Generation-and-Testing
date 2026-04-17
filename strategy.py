#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Williams %R mean reversion + volume confirmation + chop regime filter.
Long when 1d Williams %R < -80 (oversold) with volume confirmation and choppy market (CHOP > 61.8).
Short when 1d Williams %R > -20 (overbought) with volume confirmation and choppy market (CHOP > 61.8).
Williams %R identifies extreme price levels likely to revert, volume confirms participation,
and chop regime ensures we only mean-revert in ranging markets where the edge exists.
Designed to work in both bull (buy dips in range) and bear (sell rallies in range) markets.
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
    
    # Get 1d data for Williams %R and chop regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HH - C) / (HH - LL)
    lookback = 14
    highest_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Calculate 1d Choppiness Index: CHOP = 100 * log10(sum(ATR) / (log(n) * (HH - LL)))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # first period
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (np.log10(14) * (hh_14 - ll_14)))
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R and chop calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume and chop regime
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume and chop regime
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -50 (mean reversion complete)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -50 (mean reversion complete)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dWilliamsR_MeanReversion_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0