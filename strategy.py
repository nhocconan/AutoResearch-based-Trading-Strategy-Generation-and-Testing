#!/usr/bin/env python3
# 6h_Choppy_Market_Mean_Reversion
# Hypothesis: In choppy markets (high chop index), price tends to revert to the mean.
# Use daily chop index to filter for ranging conditions, then trade mean reversion at
# Bollinger Band extremes (2 std dev) with volume confirmation.
# Works in both bull and bear markets because chop index identifies ranging regimes
# that occur regardless of trend direction. Targets 20-40 trades/year to minimize fee drag.

name = "6h_Choppy_Market_Mean_Reversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for chop index and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Chop Index (14-period)
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = true_range(df_1d['high'], df_1d['low'], np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Chop Index: 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum / (atr14 * 14)) / np.log10(14)
    
    # Bollinger Bands (20, 2)
    bb_middle = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align daily indicators to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    
    # Volume confirmation (20-period MA on 6x)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for chop (34) and BB (20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(bb_middle_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop index > 61.8 indicates ranging/choppy market
        chopping = chop_aligned[i] > 61.8
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price at lower BB + chopping market + volume confirmation
            if close[i] <= bb_lower_aligned[i] and chopping and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price at upper BB + chopping market + volume confirmation
            elif close[i] >= bb_upper_aligned[i] and chopping and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches middle BB or chop ends
            if close[i] >= bb_middle_aligned[i] or not chopping:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches middle BB or chop ends
            if close[i] <= bb_middle_aligned[i] or not chopping:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals