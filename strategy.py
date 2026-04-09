#!/usr/bin/env python3
# 6h_weekly_pivot_volume_v1
# Hypothesis: 6h strategy using weekly Camarilla pivot levels (from 1w HTF) for structure,
# volume confirmation for momentum, and 1d EMA200 for trend filter.
# Weekly pivot levels provide strong support/resistance that works in both bull and bear markets.
# Volume confirmation ensures breakouts have follow-through.
# EMA200 filter ensures we trade with the higher timeframe trend.
# Discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Weekly Camarilla pivot levels (based on previous week's range)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    pivot_point = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    range_1w = prev_high_1w - prev_low_1w
    
    # Camarilla levels: H4, L4 (breakout levels) and H3, L3 (strong intraday S/R)
    h4 = pivot_point + (range_1w * 1.1 / 2)
    l4 = pivot_point - (range_1w * 1.1 / 2)
    h3 = pivot_point + (range_1w * 1.1 / 4)
    l3 = pivot_point - (range_1w * 1.1 / 4)
    
    # Align to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below H3 OR trend turns bearish
            if close[i] < h3_aligned[i] or close[i] < ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above L3 OR trend turns bullish
            if close[i] > l3_aligned[i] or close[i] > ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above H4 with bullish trend
                if close[i] > h4_aligned[i] and close[i] > ema200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below L4 with bearish trend
                elif close[i] < l4_aligned[i] and close[i] < ema200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals