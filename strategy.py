#!/usr/bin/env python3
# 6h_camarilla_pivot_1w_trend_volume_v1
# Hypothesis: Use weekly pivot levels as trend filter and daily Camarilla levels for entry.
# In uptrend (price above weekly pivot), go long at Camarilla L3 with volume confirmation.
# In downtrend (price below weekly pivot), go short at Camarilla H3 with volume confirmation.
# Weekly pivot provides structural bias, Camarilla H3/L3 offer high-probability reversal/breakout levels.
# Volume filter ensures institutional participation. Designed for low turnover: 15-35 trades/year.
# Works in bull/bear: trend filter aligns with higher timeframe, entries occur at statistically significant levels.

name = "6h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    wp_high = df_1w['high'].values
    wp_low = df_1w['low'].values
    wp_close = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    wp_pivot = (wp_high + wp_low + wp_close) / 3.0
    # Align weekly pivot to 6s timeframe (wait for weekly close)
    wp_pivot_aligned = align_htf_to_ltf(prices, df_1w, wp_pivot)
    
    # Daily data for Camarilla levels (entry signals)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using prior day's OHLC
    # H3 = C + (H - L) * 1.1 / 2
    # L3 = C - (H - L) * 1.1 / 2
    dh = df_1d['high'].values
    dl = df_1d['low'].values
    dc = df_1d['close'].values
    
    # Camarilla H3 and L3
    camarilla_h3 = dc + (dh - dl) * 1.1 / 2.0
    camarilla_l3 = dc - (dh - dl) * 1.1 / 2.0
    # Align Camarilla levels to 6s timeframe (wait for daily close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume filter: volume > 1.8x 24-period average (4 days on 6s chart)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    if n >= vol_period:
        vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(vol_period, 1) + 1
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(wp_pivot_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.8 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below weekly pivot or reverses at H3
            if close[i] < wp_pivot_aligned[i] or close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above weekly pivot or reverses at L3
            if close[i] > wp_pivot_aligned[i] or close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above weekly pivot, at L3 with volume
            if close[i] > wp_pivot_aligned[i] and \
               abs(close[i] - camarilla_l3_aligned[i]) < (camarilla_h3_aligned[i] - camarilla_l3_aligned[i]) * 0.02 and \
               volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: price below weekly pivot, at H3 with volume
            elif close[i] < wp_pivot_aligned[i] and \
                 abs(close[i] - camarilla_h3_aligned[i]) < (camarilla_h3_aligned[i] - camarilla_l3_aligned[i]) * 0.02 and \
                 volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals