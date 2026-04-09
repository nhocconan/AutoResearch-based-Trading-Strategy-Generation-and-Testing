#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_v1
# Hypothesis: 1d strategy using weekly Donchian channels with volume confirmation.
# Long when price breaks above weekly Donchian high (20-period) with volume > 1.5x 20-day average.
# Short when price breaks below weekly Donchian low (20-period) with volume > 1.5x 20-day average.
# Exit when price closes back inside weekly Donchian channels (midpoint).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture strong breakouts in both bull and bear markets while avoiding false signals.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_v1"
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
    
    # Get weekly data for Donchian channels (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: highest high over 20 periods
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over 20 periods
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low channels
    midpoint_20 = (high_20 + low_20) / 2.0
    
    # Align all levels to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    midpoint_20_aligned = align_htf_to_ltf(prices, df_1w, midpoint_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(midpoint_20_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price closes back below weekly Donchian midpoint
            if close[i] < midpoint_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes back above weekly Donchian midpoint
            if close[i] > midpoint_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation
            bullish_breakout = (close[i] > high_20_aligned[i]) and volume_confirmed
            bearish_breakout = (close[i] < low_20_aligned[i]) and volume_confirmed
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals