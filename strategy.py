#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w Donchian(20) breakout + volume confirmation + ATR stoploss.
Long when price breaks above weekly Donchian high with volume > 1.5x 20-period average.
Short when price breaks below weekly Donchian low with volume > 1.5x 20-period average.
Exit when price retraces to weekly Donchian midpoint.
Weekly Donchian provides robust structure; breakouts with volume reduce false signals.
Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag. Uses discrete sizing 0.25.
Works in both bull (trend continuation) and bear (mean reversion after volatility spikes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and volume
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    upper_20 = high_series.rolling(window=20, min_periods=20).max().values
    lower_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get weekly volume 20-period average
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1d
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(vol_ma_20_1w_aligned[i]) or np.isnan(volume_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current weekly volume > 1.5x 20-period average
        volume_confirmed = volume_1w_aligned[i] > 1.5 * vol_ma_20_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume
            if close[i] > upper_20_aligned[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume
            elif close[i] < lower_20_aligned[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Donchian midpoint
            midpoint_20 = (upper_20 + lower_20) / 2
            midpoint_20_aligned = align_htf_to_ltf(prices, df_1w, midpoint_20)
            if close[i] < midpoint_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Donchian midpoint
            midpoint_20 = (upper_20 + lower_20) / 2
            midpoint_20_aligned = align_htf_to_ltf(prices, df_1w, midpoint_20)
            if close[i] > midpoint_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wDonchian20_Volume"
timeframe = "1d"
leverage = 1.0