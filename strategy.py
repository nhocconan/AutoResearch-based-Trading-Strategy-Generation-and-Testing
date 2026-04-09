#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_breakout_v1
# Hypothesis: 6h strategy combining weekly Camarilla pivot levels with 6h Donchian breakout and volume confirmation.
# Long: Price breaks above 6h Donchian(20) high AND weekly R4 level with volume > 1.8x 20-period average.
# Short: Price breaks below 6h Donchian(20) low AND weekly S4 level with volume > 1.8x 20-period average.
# Exit: Price returns to weekly pivot point (PP) or breaks opposite Donchian level.
# Uses weekly Camarilla for key support/resistance (HTF), 6h for execution (Donchian breakout), volume for confirmation.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_breakout_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for Camarilla pivot levels (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r4 = close_1w + range_1w * 1.1 / 2.0
    s4 = close_1w - range_1w * 1.1 / 2.0
    
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # 6h Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to weekly pivot or breaks below 6h Donchian low
            if close[i] <= pivot_aligned[i] or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to weekly pivot or breaks above 6h Donchian high
            if close[i] >= pivot_aligned[i] or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation: price must break both 6h Donchian AND weekly Camarilla levels
            bullish_breakout = (close[i] > donchian_high[i]) and (close[i] > r4_aligned[i]) and volume_confirmed
            bearish_breakout = (close[i] < donchian_low[i]) and (close[i] < s4_aligned[i]) and volume_confirmed
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals