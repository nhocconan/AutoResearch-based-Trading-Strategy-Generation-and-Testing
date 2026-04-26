#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend_VolumeSpike
Hypothesis: 6h Williams Fractal breakouts with 1d trend filter (EMA50) and volume spike confirmation.
Enters long on bullish fractal breakout above R1 with bullish 1d trend (price > EMA50) and volume spike (>2.0x 20 EMA volume).
Enters short on bearish fractal breakout below S1 with bearish 1d trend (price < EMA50) and volume spike.
Exits when price crosses the 1d EMA50 (trend reversal).
Designed for 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
Uses discrete position sizing (0.25) to minimize churn. Works in both bull and bear markets by following 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on 1d (need 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align fractals with 2-bar additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 1d EMA50 + fractal calculation)
    start_idx = 50 + 2  # EMA50 + 2-bar fractal confirmation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: bullish fractal breakout + bullish 1d trend + volume spike
        if (bullish_fractal_aligned[i] and 
            close[i] > ema_50_1d_aligned[i] and 
            volume_spike[i]):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: bearish fractal breakout + bearish 1d trend + volume spike
        elif (bearish_fractal_aligned[i] and 
              close[i] < ema_50_1d_aligned[i] and 
              volume_spike[i]):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses 1d EMA50 (trend reversal)
        elif position == 1 and close[i] < ema_50_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_50_1d_aligned[i]:
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

name = "6h_WilliamsFractal_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0