#!/usr/bin/env python3
# 1d_weekly_higher_high_lower_low_volume_v1
# Hypothesis: 1d strategy using weekly higher high/lower low structure with volume confirmation.
# Long: Price makes weekly higher high (close > weekly high) with volume > 1.5x 20-day average.
# Short: Price makes weekly lower low (close < weekly low) with volume > 1.5x 20-day average.
# Exit: Price crosses weekly pivot point (PP) in opposite direction.
# Uses weekly structure from 1w timeframe as trend filter. Volume confirms breakouts.
# Target: 7-25 trades/year (30-100 total over 4 years). Works in bull/bear via weekly structure.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_higher_high_lower_low_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for weekly structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly higher high / lower low
    # Higher high: current weekly high > previous weekly high
    hh_1w = np.zeros(len(df_1w), dtype=bool)
    ll_1w = np.zeros(len(df_1w), dtype=bool)
    hh_1w[1:] = high_1w[1:] > high_1w[:-1]
    ll_1w[1:] = low_1w[1:] < low_1w[:-1]
    
    # Weekly pivot point (PP) = (High + Low + Close) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly structure to daily
    hh_aligned = align_htf_to_ltf(prices, df_1w, hh_1w.astype(float))
    ll_aligned = align_htf_to_ltf(prices, df_1w, ll_1w.astype(float))
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(hh_aligned[i]) or np.isnan(ll_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price crosses below weekly PP
            if close[i] < pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above weekly PP
            if close[i] > pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Weekly higher high with volume confirmation
            if hh_aligned[i] > 0.5 and volume_confirmed:  # Boolean as float > 0.5
                position = 1
                signals[i] = 0.25
            # Short entry: Weekly lower low with volume confirmation
            elif ll_aligned[i] > 0.5 and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals