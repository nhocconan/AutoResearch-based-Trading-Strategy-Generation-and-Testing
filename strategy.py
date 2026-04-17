#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Williams %R mean reversion + volume confirmation + chop regime filter.
Long when Williams %R < -80 (oversold) with volume confirmation and choppy market (CHOP > 61.8).
Short when Williams %R > -20 (overbought) with volume confirmation and choppy market (CHOP > 61.8).
Exit when Williams %R returns to -50 (mean reversion) or chop regime ends (CHOP < 38.2).
Williams %R identifies extreme price levels for mean reversion in ranging markets.
Chop filter ensures we only trade in ranging conditions where mean reversion works.
Volume confirmation avoids false signals in low participation periods.
Designed to work in both bull (2021-2024) and bear (2025+) markets by focusing on ranging regimes.
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
    
    # Get 1d data for Williams %R and chop calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Calculate 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(high_1d) - pd.Series(low_1d).shift(1)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(close_1d)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R, chop, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume confirmation and chop regime
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                chop_regime):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume confirmation and chop regime
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  chop_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to mean (-50) OR chop regime ends (trending market)
            if (williams_r_aligned[i] >= -50 or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to mean (-50) OR chop regime ends (trending market)
            if (williams_r_aligned[i] <= -50 or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dWilliamsR_MeanReversion_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0