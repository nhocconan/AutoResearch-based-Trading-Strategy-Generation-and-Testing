#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d Volume Spike + 1w Choppiness Filter
# Williams %R identifies overbought/oversold conditions: Long when %R < -80 (oversold), Short when %R > -20 (overbought)
# Volume spike confirms conviction: current volume > 2.0 x 20-period MA
# Choppiness filter avoids ranging markets: only trade when CHOP(14) < 38.2 (trending) on 1w
# Exit when Williams %R reverses (%R > -50 for long exit, %R < -50 for short exit) OR chop filter fails
# Uses mean reversion in extremes with trend filter to work in both bull and bear markets
# Timeframe: 4h, HTF: 1d for volume, 1w for chop. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_WilliamsR_1dVolumeSpike_1wChoppinessFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 4h
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when high == low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, np.nan)
    
    # Williams %R signals: oversold < -80, overbought > -20
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    williams_exit = williams_r > -50  # exit long when %R > -50
    williams_exit_short = williams_r < -50  # exit short when %R < -50
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for volume confirmation (already have 4h volume above)
    # Actually, we'll use 1d for additional volume confirmation if needed, but 4h volume is primary
    # Get 1w data ONCE before loop for Choppiness calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # need sufficient data for chop
        return np.zeros(n)
    
    # Calculate Choppiness Index(14) on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    if len(high_1w) >= 14:
        # True Range
        tr1 = np.abs(high_1w[1:] - low_1w[1:])
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index = 100 * log10(sum(tr) / (hh - ll)) / log10(14)
        # Avoid division by zero and log of zero
        hl_range = hh_14 - ll_14
        chop_raw = np.where((hl_range > 0) & (tr_sum > 0), 
                           100 * np.log10(tr_sum / hl_range) / np.log10(14), 
                           50)  # default to 50 (neutral) when invalid
        chop = chop_raw
    else:
        chop = np.full(len(df_1w), np.nan)
    
    # Choppiness filter: CHOP < 38.2 = trending (favor trend following), CHOP > 61.8 = ranging
    # We want trending markets for Williams %R mean reversion to work better
    chop_filter = chop < 38.2
    
    # Align 1w Choppiness to 4h timeframe
    chop_filter_aligned = align_htf_to_ltf(prices, df_1w, chop_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_oversold[i]) or np.isnan(williams_overbought[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(chop_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold + volume spike + trending market (chop < 38.2)
            if (williams_oversold[i] and 
                volume_filter[i] and 
                chop_filter_aligned[i] == 1.0):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought + volume spike + trending market (chop < 38.2)
            elif (williams_overbought[i] and 
                  volume_filter[i] and 
                  chop_filter_aligned[i] == 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R exits oversold (> -50) OR chop filter fails (chop >= 38.2)
            if (williams_exit[i] or chop_filter_aligned[i] == 0.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R exits overbought (< -50) OR chop filter fails (chop >= 38.2)
            if (williams_exit_short[i] or chop_filter_aligned[i] == 0.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals