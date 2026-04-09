#!/usr/bin/env python3
# 6h_donchian_breakout_weekly_pivot_volume_v1
# Hypothesis: 6h Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and weekly HTF pivot direction filter. Enters long when price breaks above Donchian upper band with volume confirmation and price above weekly pivot; short when price breaks below Donchian lower band with volume confirmation and price below weekly pivot. Exits on opposite Donchian band touch. Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 50-150 total trades over 4 years) to work in both bull and bear markets by following institutional volume-driven breakouts in alignment with higher timeframe pivot levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume_v1"
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
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    
    # Weekly OHLC for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot point (standard: (H+L+C)/3)
    prev_weekly_high = df_1w['high'].values
    prev_weekly_low = df_1w['low'].values
    prev_weekly_close = df_1w['close'].values
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches or breaks Donchian lower band
            if close[i] <= donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks Donchian upper band
            if close[i] >= donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and weekly pivot alignment
            if volume_confirmed:
                # Long: price breaks above Donchian upper band with volume and price above weekly pivot
                if close[i] > donchian_upper[i] and close[i] > weekly_pivot_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower band with volume and price below weekly pivot
                elif close[i] < donchian_lower[i] and close[i] < weekly_pivot_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals