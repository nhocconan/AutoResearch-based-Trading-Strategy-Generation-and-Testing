#!/usr/bin/env python3
# 6h_donchian_1w_pivot_volume_v1
# Hypothesis: 6h Donchian(20) breakout with weekly Camarilla pivot direction filter and volume confirmation.
# Weekly Camarilla levels (H3/L3) determine trend bias: long only above weekly H3, short only below weekly L3.
# 6h Donchian breakout provides entry timing in direction of weekly trend.
# Volume > 1.5x 20-period average confirms institutional participation.
# Works in bull/bear by aligning with weekly structure. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 12-25 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_1w_pivot_volume_v1"
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
    
    # 1w HTF data for Camarilla pivots (weekly trend bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Camarilla pivot levels (based on previous week's range)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    pivot_point = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    range_1w = prev_high_1w - prev_low_1w
    
    # Weekly Camarilla H3 and L3 levels (strongest support/resistance)
    h3_1w = pivot_point + (range_1w * 1.1 / 4)
    l3_1w = pivot_point - (range_1w * 1.1 / 4)
    
    # Align weekly Camarilla levels to 6h timeframe
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    
    # 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(h3_1w_aligned[i]) or 
            np.isnan(l3_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly H3 OR 6h Donchian lower band
            if close[i] < h3_1w_aligned[i] or close[i] < lowest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly L3 OR 6h Donchian upper band
            if close[i] > l3_1w_aligned[i] or close[i] > highest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above 6h Donchian upper band AND above weekly H3 (bullish bias)
                if close[i] > highest_20[i] and close[i] > h3_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below 6h Donchian lower band AND below weekly L3 (bearish bias)
                elif close[i] < lowest_20[i] and close[i] < l3_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals