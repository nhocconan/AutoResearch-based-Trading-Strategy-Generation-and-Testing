#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 1d volume regime filter and weekly pivot direction
# Donchian breakouts capture momentum in both bull and bear markets
# 1d volume regime (high/low volume) filters false breakouts
# Weekly pivot provides directional bias from higher timeframe
# Works in trending and ranging markets by adapting to volume conditions
# Uses discrete position sizing (0.25) to minimize fee churn

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1d volume regime: 20-period average
    vol_avg_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20d)
    
    # Weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivots = (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3
    ranges = high_1w[:-1] - low_1w[:-1]
    r1 = 2 * pivots - low_1w[:-1]
    s1 = 2 * pivots - high_1w[:-1]
    r2 = pivots + ranges
    s2 = pivots - ranges
    r3 = high_1w[:-1] + 2 * (pivots - low_1w[:-1])
    s3 = low_1w[:-1] - 2 * (high_1w[:-1] - pivots)
    
    # Align weekly data to 6h
    pivots_aligned = align_htf_to_ltf(prices, df_1w, pivots)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Donchian channel (20-period) on 6h
    donchian_len = 20
    dc_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    dc_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    dc_mid = (dc_high + dc_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(donchian_len, n):
        # Skip if data not ready
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(vol_avg_20d_aligned[i]) or np.isnan(pivots_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above Donchian high + volume expansion + above weekly pivot
            breakout_up = close[i] > dc_high[i-1]  # breakout confirmed on close
            volume_expansion = volume[i] > 1.5 * vol_avg_20d_aligned[i]
            above_pivot = close[i] > pivots_aligned[i]
            
            # Short conditions: breakdown below Donchian low + volume expansion + below weekly pivot
            breakout_down = close[i] < dc_low[i-1]  # breakdown confirmed on close
            below_pivot = close[i] < pivots_aligned[i]
            
            if breakout_up and volume_expansion and above_pivot:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_expansion and below_pivot:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: reverse breakout or volume contraction
            if position == 1:
                # Exit long: breakdown below Donchian low or volume contraction
                breakout_down = close[i] < dc_low[i-1]
                volume_contraction = volume[i] < 0.5 * vol_avg_20d_aligned[i]
                if breakout_down or volume_contraction:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: breakout above Donchian high or volume contraction
                breakout_up = close[i] > dc_high[i-1]
                volume_contraction = volume[i] < 0.5 * vol_avg_20d_aligned[i]
                if breakout_up or volume_contraction:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dVolumeRegime_1wPivot"
timeframe = "6h"
leverage = 1.0