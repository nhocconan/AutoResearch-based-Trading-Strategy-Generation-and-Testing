#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian channel breakout with weekly pivot direction and volume confirmation
    # Works in both bull and bear markets: breakouts from price channels with weekly structure
    # Weekly pivot provides directional bias, volume confirms breakout strength
    # Target: 12-37 trades/year (50-150 over 4 years)
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point calculation (standard formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Weekly trend: above R1 = bullish, below S1 = bearish
    weekly_bullish = close_1w > r1_1w
    weekly_bearish = close_1w < s1_1w
    
    # Align weekly trend to 6h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # 6h Donchian channel (20 periods)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band with weekly bullish bias and volume surge
            if close[i] > donchian_high[i] and weekly_bullish_aligned[i] > 0.5 and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower band with weekly bearish bias and volume surge
            elif close[i] < donchian_low[i] and weekly_bearish_aligned[i] > 0.5 and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian band or weekly pivot
            if position == 1:
                if close[i] < donchian_low[i] or close[i] < pivot_1w[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high[i] or close[i] > pivot_1w[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0