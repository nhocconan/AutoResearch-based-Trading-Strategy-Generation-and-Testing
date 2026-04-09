#!/usr/bin/env python3
# 12h_camarilla_1d_trend_volume_v1
# Hypothesis: 12h strategy using daily Camarilla pivot levels (H3/L3) for mean reversion, 
# filtered by 1d trend (price above/below weekly midpoint) and volume confirmation (>1.5x 20-period MA).
# In ranging markets (chop > 61.8), price tends to revert to pivot point from H3/L3 levels.
# Volume confirms institutional participation. Discrete sizing (±0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Camarilla pivot levels (based on previous day's range)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels: H3 (strong resistance), L3 (strong support)
    h3 = pivot_point + (range_1d * 1.1 / 4)
    l3 = pivot_point - (range_1d * 1.1 / 4)
    pivot_point_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # 1w HTF data for trend filter (weekly midpoint)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    weekly_midpoint = (highest_high_20 + lowest_low_20) / 2
    weekly_midpoint_aligned = align_htf_to_ltf(prices, df_1w, weekly_midpoint)
    
    # Choppiness index regime filter (1d)
    # Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending
    atr_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * np.sqrt(14) / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pivot_point_aligned[i]) or
            np.isnan(weekly_midpoint_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches pivot point (mean reversion target) OR weekly bearish bias
            if close[i] >= pivot_point_aligned[i] or close[i] < weekly_midpoint_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches pivot point (mean reversion target) OR weekly bullish bias
            if close[i] <= pivot_point_aligned[i] or close[i] > weekly_midpoint_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and ranging regime (chop > 61.8)
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            ranging_regime = chop_aligned[i] > 61.8
            
            if volume_confirmed and ranging_regime:
                # Long: price at L3 support with weekly bullish bias
                if close[i] <= l3_aligned[i] and close[i] > weekly_midpoint_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price at H3 resistance with weekly bearish bias
                elif close[i] >= h3_aligned[i] and close[i] < weekly_midpoint_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals