#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_volume_v1
# Hypothesis: 6h strategy using weekly pivot levels for regime filter and daily Donchian breakout with volume confirmation.
# Long: Weekly close above weekly pivot (bullish regime) + price breaks above daily Donchian upper (20) + volume > 1.5x 20-period average.
# Short: Weekly close below weekly pivot (bearish regime) + price breaks below daily Donchian lower (20) + volume > 1.5x 20-period average.
# Exit: Price returns to opposite Donchian level (exit long at lower band, exit short at upper band).
# Uses 6h primary timeframe with 1d HTF for Donchian levels and 1w HTF for weekly pivot regime.
# Designed for low trade frequency (~15-35/year) to minimize fee drag while capturing strong directional moves.
# Works in bull markets via breakouts in bullish regime and bear markets via shorting in bearish regime.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_volume_v1"
timeframe = "6h"
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
    
    # Get 1d data for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian channels (20-period)
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 6h
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Get 1w data for weekly pivot regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align 1w weekly pivot to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to daily Donchian lower band
            if close[i] <= donchian_low_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to daily Donchian upper band
            if close[i] >= donchian_high_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Weekly close above weekly pivot (bullish regime) + break above daily Donchian upper + volume
            if (close_1w[-1] > weekly_pivot_aligned[i] and    # Weekly close above pivot (bullish regime)
                close[i] > donchian_high_1d_aligned[i] and    # Break above daily Donchian upper
                volume_confirmed):                            # Volume spike
                position = 1
                signals[i] = 0.25
            # Short entry: Weekly close below weekly pivot (bearish regime) + break below daily Donchian lower + volume
            elif (close_1w[-1] < weekly_pivot_aligned[i] and  # Weekly close below pivot (bearish regime)
                  close[i] < donchian_low_1d_aligned[i] and   # Break below daily Donchian lower
                  volume_confirmed):                          # Volume spike
                position = -1
                signals[i] = -0.25
    
    return signals