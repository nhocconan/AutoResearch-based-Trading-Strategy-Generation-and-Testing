#!/usr/bin/env python3
# 6h_Donchian20_WeeklyPivot_Direction_Volume
# Hypothesis: 6h Donchian(20) breakout combined with weekly pivot direction (1w high/low) and volume confirmation.
# In bull markets: buy breakouts above weekly pivot resistance; in bear markets: sell breakdowns below weekly pivot support.
# Weekly pivot provides structural context, Donchian captures breakouts, volume confirms strength.
# Targets 15-35 trades/year to minimize fee drag while capturing major moves.

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H + L + C) / 3
    # Then derive support/resistance levels
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_r1 = 2 * weekly_pivot - df_1w['low']
    weekly_s1 = 2 * weekly_pivot - df_1w['high']
    weekly_r2 = weekly_pivot + (weekly_r1 - weekly_s1)
    weekly_s2 = weekly_pivot - (weekly_r1 - weekly_s1)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1.values)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1.values)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2.values)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2.values)
    
    # Calculate Donchian channels (20-period)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market bias using weekly pivot
        # Above weekly R1 = bullish bias, below weekly S1 = bearish bias
        bullish_bias = close[i] > weekly_r1_aligned[i]
        bearish_bias = close[i] < weekly_s1_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Donchian breakout signals
        breakout_up = high[i] > donchian_upper[i-1]  # Break above previous upper band
        breakdown_down = low[i] < donchian_lower[i-1]  # Break below previous lower band
        
        if position == 0:
            # Long entry: bullish bias + upward breakout + volume confirmation
            if bullish_bias and breakout_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish bias + downward breakdown + volume confirmation
            elif bearish_bias and breakdown_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly pivot or Donchian lower band
            if close[i] < weekly_pivot_aligned[i] or low[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly pivot or Donchian upper band
            if close[i] > weekly_pivot_aligned[i] or high[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals