#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation.
# Donchian provides clear breakout levels, weekly pivot gives directional bias from higher timeframe,
# volume confirms conviction. Designed for low trade frequency (15-35/year) to minimize fee drag.
# Works in bull markets (break above upper band with bullish weekly pivot) and bear markets 
# (break below lower band with bearish weekly pivot).
name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
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
    
    # Get daily data for pivot calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for pivot calculation (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using previous week's data)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pp = (high_w + low_w + close_w) / 3
    # Resistance 1 = (2 * PP) - L
    r1 = (2 * pp) - low_w
    # Support 1 = (2 * PP) - H
    s1 = (2 * pp) - high_w
    # Resistance 2 = PP + (H - L)
    r2 = pp + (high_w - low_w)
    # Support 2 = PP - (H - L)
    s2 = pp - (high_w - low_w)
    # Resistance 3 = H + 2*(PP - L)
    r3 = high_w + 2 * (pp - low_w)
    # Support 3 = L - 2*(H - PP)
    s3 = low_w - 2 * (high_w - pp)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 20-period Donchian channels (using previous period's data)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND weekly pivot bias bullish (price > PP) AND volume
            if (close[i] > high_20[i] and 
                close[i] > pp_aligned[i] and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND weekly pivot bias bearish (price < PP) AND volume
            elif (close[i] < low_20[i] and 
                  close[i] < pp_aligned[i] and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower Donchian OR weekly pivot turns bearish (price < S1)
            if (close[i] < low_20[i] or 
                close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper Donchian OR weekly pivot turns bullish (price > R1)
            if (close[i] > high_20[i] or 
                close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals