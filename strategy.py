#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with Volume Confirmation and 1w Trend Filter
# Uses Williams Alligator (3 SMAs: Jaw 13, Teeth 8, Lips 5) on daily timeframe.
# Long when Lips > Teeth > Jaw (bullish alignment) + volume above average + weekly close > weekly open (bullish weekly).
# Short when Lips < Teeth < Jaw (bearish alignment) + volume above average + weekly close < weekly open (bearish weekly).
# Exit when Alligator lines cross in opposite direction or weekly trend changes.
# Designed to catch sustained trends while avoiding whipsaws in ranging markets.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation reduces false breakouts.
# Target: 20-60 total trades over 4 years (5-15/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Williams Alligator on 1d: SMAs of median price (HL/2)
    median_price_1d = (high_1d + low_1d) / 2
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values  # SMA5
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values   # SMA8
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values  # SMA13
    
    # Weekly trend: bullish if close > open, bearish if close < open
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align Alligator lines and weekly trend to 1d timeframe
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):  # Start after Alligator warmup
        # Skip if any required data is NaN
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            continue
        
        # Volume condition: current volume > 1.5x 20-day median volume
        vol_cond = volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1])
        
        # Long entry: Bullish Alligator alignment + volume + weekly bullish
        if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and
            vol_cond and
            weekly_bullish_aligned[i] > 0.5 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bearish Alligator alignment + volume + weekly bearish
        elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and
              vol_cond and
              weekly_bearish_aligned[i] > 0.5 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Alligator crosses in opposite direction or weekly trend changes
        elif position == 1 and (lips_aligned[i] < jaw_aligned[i] or weekly_bullish_aligned[i] < 0.5):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (lips_aligned[i] > jaw_aligned[i] or weekly_bearish_aligned[i] < 0.5):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_WilliamsAlligator_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0