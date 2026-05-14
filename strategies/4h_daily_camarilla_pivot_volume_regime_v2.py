#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_regime_v2
# Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume confirmation and choppiness regime filter.
# Long: Price breaks above H3 pivot with volume > 1.5x 20-period average AND chop > 61.8 (ranging market)
# Short: Price breaks below L3 pivot with volume > 1.5x 20-period average AND chop > 61.8 (ranging market)
# Exit: Price returns to H4/L4 levels or opposite pivot break
# Uses 4h primary timeframe with 1d HTF for Camarilla pivot calculation and choppiness filter.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivots and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (High + Low + Close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    # Camarilla levels
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    h4_1d = pivot_1d + (range_1d * 1.1 / 2)
    l4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 4h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Calculate choppiness index on 1d (14-period)
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr_sum = np.zeros_like(close_arr)
        true_range = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            tr = max(high_arr[i] - low_arr[i], 
                     abs(high_arr[i] - close_arr[i-1]),
                     abs(low_arr[i] - close_arr[i-1]))
            true_range[i] = tr
        # Calculate ATR using Wilder's smoothing (equivalent to RMA)
        atr = np.zeros_like(close_arr)
        atr[period] = np.mean(true_range[1:period+1]) if period < len(true_range) else 0
        for i in range(period+1, len(close_arr)):
            atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
        # Sum ATR over period
        atr_sum = np.zeros_like(close_arr)
        for i in range(period, len(close_arr)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        # Calculate choppiness
        chop = np.zeros_like(close_arr)
        max_high = np.zeros_like(close_arr)
        min_low = np.zeros_like(close_arr)
        for i in range(period, len(close_arr)):
            max_high[i] = np.max(high_arr[i-period+1:i+1])
            min_low[i] = np.min(low_arr[i-period+1:i+1])
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral when no range
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price returns to H4 level or breaks below L3 (opposite signal)
            if close[i] <= h4_1d_aligned[i] or close[i] < l3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to L4 level or breaks above H3 (opposite signal)
            if close[i] >= l4_1d_aligned[i] or close[i] > h3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above H3 with volume confirmation in ranging market
            if close[i] > h3_1d_aligned[i] and volume_confirmed and chop_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 with volume confirmation in ranging market
            elif close[i] < l3_1d_aligned[i] and volume_confirmed and chop_filter:
                position = -1
                signals[i] = -0.25
    
    return signals