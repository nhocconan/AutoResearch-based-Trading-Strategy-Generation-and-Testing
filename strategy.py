#!/usr/bin/env python3
"""
1d Donchian Breakout with Volume Spike and 1-week Trend Filter
Strategy: Enter long when price breaks above weekly Donchian high with volume
          and price > daily close (bullish bias). Enter short when price breaks
          below weekly Donchian low with volume and price < daily close.
          Exit when price returns to weekly midpoint.
          Uses weekly Donchian channels for trend and daily close for bias.
          Designed for low trade frequency with clear breakout edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly Donchian high and low (20-period)
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align weekly levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        mid = donchian_mid_aligned[i]
        
        if position == 0:
            # Long: break above weekly Donchian high with volume spike and above weekly close (bullish bias)
            if (price > upper and volume_spike[i] and price > weekly_close[-1] if len(weekly_close) > 0 else True):
                # Simplified: use price > weekly close as bias, but since we don't have weekly close aligned,
                # use price > open as simple bullish bias (price above open suggests buying pressure)
                if price > prices['open'].iloc[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: break below weekly Donchian low with volume spike and below weekly close (bearish bias)
            elif (price < lower and volume_spike[i] and price < weekly_close[-1] if len(weekly_close) > 0 else True):
                if price < prices['open'].iloc[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price returns to weekly midpoint
            if price <= mid:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price returns to weekly midpoint
            if price >= mid:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian_Breakout_VolumeSpike_WeeklyTrend"
timeframe = "1d"
leverage = 1.0