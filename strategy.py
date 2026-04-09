#!/usr/bin/env python3
# 4h_1d_1w_camarilla_pivot_volume_v1
# Hypothesis: 4h strategy combining 1w HTF trend filter (price above/below weekly pivot) with 1d Camarilla breakout entries.
# Long: Weekly bullish trend (close > weekly pivot) + price breaks above 1d H4 with volume > 2.0x 20-period average.
# Short: Weekly bearish trend (close < weekly pivot) + price breaks below 1d L4 with volume > 2.0x 20-period average.
# Exit: Price returns to opposite Camarilla level (H3 for longs, L3 for shorts) or weekly pivot flip.
# Uses 4h primary timeframe with 1w HTF for trend and 1d HTF for Camarilla levels.
# Designed for low trade frequency (~20-50/year) to minimize fee drag while capturing institutional breakouts.
# Works in bull markets via breakouts and bear markets via fade-from-extremes logic at weekly pivot.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_camarilla_pivot_volume_v1"
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
    open_prices = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter (weekly pivot)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: (High + Low + Close) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Align weekly pivot to 4h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    h4_1d = pivot_1d + (range_1d * 1.1 / 2)
    l4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 4h
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(open_prices[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        # Bullish candle: close > open
        bullish_candle = close[i] > open_prices[i]
        # Bearish candle: close < open
        bearish_candle = close[i] < open_prices[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to H3 or weekly trend turns bearish
            if close[i] <= h3_1d_aligned[i] or close[i] < pivot_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to L3 or weekly trend turns bullish
            if close[i] >= l3_1d_aligned[i] or close[i] > pivot_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Weekly bullish trend + price breaks above H4 with volume and bullish candle
            if (close[i] > pivot_1w_aligned[i] and  # Weekly bullish trend
                close[i] > h4_1d_aligned[i] and    # Break above H4
                volume_confirmed and               # Volume spike
                bullish_candle):                   # Bullish candle
                position = 1
                signals[i] = 0.25
            # Short entry: Weekly bearish trend + price breaks below L4 with volume and bearish candle
            elif (close[i] < pivot_1w_aligned[i] and  # Weekly bearish trend
                  close[i] < l4_1d_aligned[i] and     # Break below L4
                  volume_confirmed and                # Volume spike
                  bearish_candle):                    # Bearish candle
                position = -1
                signals[i] = -0.25
    
    return signals