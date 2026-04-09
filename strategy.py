#!/usr/bin/env python3
# 1d_weekly_higher_high_lower_low_volume_v2
# Hypothesis: 1d strategy using weekly higher highs and lower lows with volume confirmation.
# Long: Price makes a weekly higher high (close > weekly high_1w) with volume > 1.5x 20-day average.
# Short: Price makes a weekly lower low (close < weekly low_1w) with volume > 1.5x 20-day average.
# Exit: Price crosses the weekly midpoint (average of weekly high and low).
# Uses weekly structure to capture multi-day momentum while avoiding intra-day noise.
# Volume confirmation ensures breakouts have participation.
# Target: 7-25 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_higher_high_lower_low_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly higher highs and lower lows
    # Higher high: current weekly close > previous weekly high
    hh_1w = np.zeros(len(df_1w), dtype=bool)
    ll_1w = np.zeros(len(df_1w), dtype=bool)
    hh_1w[1:] = close_1w[1:] > high_1w[:-1]
    ll_1w[1:] = close_1w[1:] < low_1w[:-1]
    
    # Weekly midpoint for exit
    midpoint_1w = (high_1w + low_1w) / 2.0
    
    # Align weekly signals to daily
    hh_aligned = align_htf_to_ltf(prices, df_1w, hh_1w.astype(float))
    ll_aligned = align_htf_to_ltf(prices, df_1w, ll_1w.astype(float))
    midpoint_aligned = align_htf_to_ltf(prices, df_1w, midpoint_1w)
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if any required data is NaN
        if (np.isnan(hh_aligned[i]) or np.isnan(ll_aligned[i]) or np.isnan(midpoint_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(high_1w_aligned[i]) or np.isnan(low_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price crosses below weekly midpoint
            if close[i] < midpoint_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above weekly midpoint
            if close[i] > midpoint_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Weekly higher high with volume confirmation
            if hh_aligned[i] > 0.5 and volume_confirmed:  # True hh signal
                position = 1
                signals[i] = 0.25
            # Short entry: Weekly lower low with volume confirmation
            elif ll_aligned[i] > 0.5 and volume_confirmed:  # True ll signal
                position = -1
                signals[i] = -0.25
    
    return signals