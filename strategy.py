#!/usr/bin/env python3
# 12h_weekly_donchian_breakout_volume_v1
# Hypothesis: 12h strategy using weekly Donchian channel breakouts with volume confirmation.
# Long: Price breaks above weekly Donchian high (20-period) with volume > 1.3x 20-period average.
# Short: Price breaks below weekly Donchian low (20-period) with volume > 1.3x 20-period average.
# Exit: Price returns to weekly midpoint (average of Donchian high/low).
# Uses weekly structure for major trend, 12h for execution, volume to avoid false breakouts.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_weekly_donchian_breakout_volume_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for Donchian channels (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: rolling max of highs
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of lows
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Weekly midpoint: average of Donchian high/low
    weekly_midpoint = (donchian_high + donchian_low) / 2.0
    
    # Align HTF indicators to LTF (12h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    weekly_midpoint_aligned = align_htf_to_ltf(prices, df_1w, weekly_midpoint)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_midpoint_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to weekly midpoint
            if close[i] <= weekly_midpoint_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to weekly midpoint
            if close[i] >= weekly_midpoint_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation
            bullish_breakout = (close[i] > donchian_high_aligned[i]) and volume_confirmed
            bearish_breakout = (close[i] < donchian_low_aligned[i]) and volume_confirmed
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals