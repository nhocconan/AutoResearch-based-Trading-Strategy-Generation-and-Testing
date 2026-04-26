#!/usr/bin/env python3
"""
12h_WilliamsFractal_Breakout_WeeklyTrend_VolumeFilter
Hypothesis: 12-hour Williams fractal breakouts with weekly trend filter and volume confirmation.
Enters long on bullish fractal breakout above weekly EMA50 with volume spike.
Enters short on bearish fractal breakout below weekly EMA50 with volume filter.
Uses Williams fractals (lagging HTF indicator requiring 2-bar confirmation delay) + weekly EMA trend.
Designed for 50-150 total trades over 4 years with discrete position sizing (0.0, ±0.25) to minimize fee drag.
Works in bull markets via long fractal breaks and bear markets via short fractal breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for weekly EMA and fractals
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Williams fractals on weekly timeframe (lagging indicator - needs 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Align with 2-bar additional delay for fractal confirmation (needs 2 future weekly bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.8 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period EMA + fractal calculation)
    start_idx = 50 + 5  # 50 for EMA + ~5 for fractal lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: bullish fractal breakout + above weekly EMA50 + volume spike
        if (not np.isnan(bullish_fractal_aligned[i]) and 
            close[i] > bullish_fractal_aligned[i] and 
            close[i] > ema_50_1w_aligned[i] and 
            volume_spike[i]):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: bearish fractal breakout + below weekly EMA50 + volume spike
        elif (not np.isnan(bearish_fractal_aligned[i]) and 
              close[i] < bearish_fractal_aligned[i] and 
              close[i] < ema_50_1w_aligned[i] and 
              volume_spike[i]):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite fractal level
        elif position == 1 and not np.isnan(bearish_fractal_aligned[i]) and close[i] < bearish_fractal_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not np.isnan(bullish_fractal_aligned[i]) and close[i] > bullish_fractal_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_WilliamsFractal_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0