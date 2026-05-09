#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ChoppinessIndex_Filter_CamarillaBreakout"
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
    
    # Get weekly data for choppiness index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly Choppiness Index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = np.zeros(len(close_1w))
    tr_1w = np.maximum(np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1])), np.abs(low_1w[1:] - close_1w[:-1]))
    tr_1w = np.concatenate([[np.inf], tr_1w])  # first TR is high-low
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr14) / (max(high14) - min(low14))) / log10(14)
    sum_atr_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr_14 / (highest_high_1w - lowest_low_1w)) / np.log10(14)
    
    # Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (trend follow)
    chop_threshold_high = 61.8
    chop_threshold_low = 38.2
    
    # Daily Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    camarilla_high = close_1d + 1.1 * range_1d / 12  # R3 level
    camarilla_low = close_1d - 1.1 * range_1d / 12   # S3 level
    
    # Daily volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    chop_12h = align_htf_to_ltf(prices, df_1w, chop)
    camarilla_high_12h = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_12h = align_htf_to_ltf(prices, df_1d, camarilla_low)
    vol_avg_1d_12h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(chop_12h[i]) or np.isnan(camarilla_high_12h[i]) or 
            np.isnan(camarilla_low_12h[i]) or np.isnan(vol_avg_1d_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_12h[i]
        resistance = camarilla_high_12h[i]
        support = camarilla_low_12h[i]
        vol_avg = vol_avg_1d_12h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        # Determine market regime
        is_ranging = chop_val > chop_threshold_high
        is_trending = chop_val < chop_threshold_low
        
        if position == 0:
            # In ranging market: mean reversion at Camarilla extremes
            if is_ranging and vol_ok:
                # Long at S3 (support)
                if close[i] < support:
                    signals[i] = 0.25
                    position = 1
                # Short at R3 (resistance)
                elif close[i] > resistance:
                    signals[i] = -0.25
                    position = -1
            # In trending market: breakout in direction of trend
            elif is_trending and vol_ok:
                # Determine trend direction from price action
                if close[i] > resistance and close[i] > close[i-1]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < support and close[i] < close[i-1]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price reaches opposite Camarilla level or chop signals trend change
            if close[i] > resistance or (chop_val < chop_threshold_low and close[i] < close[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches opposite Camarilla level or chop signals trend change
            if close[i] < support or (chop_val < chop_threshold_low and close[i] > close[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals