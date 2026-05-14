#!/usr/bin/env python3
"""
6h_WilliamsFractal_1dTrend_VolumeBreakout
Hypothesis: 6h Williams Fractal breakouts confirmed by 1d trend (price >/< EMA50) and volume spikes (>2.0x 20-bar avg). 
Enters long on bullish fractal break above resistance with volume in 1d uptrend, short on bearish fractal break below support with volume in 1d downtrend. 
Williams Fractals provide natural support/resistance levels; breakouts with volume and trend alignment capture momentum moves in both bull and bear markets. 
Designed for 6h timeframe with ~15-35 trades/year, avoiding overtrading via strict fractal confirmation and volume filters.
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
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Fractals on 1d timeframe (needs 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar delay for fractal confirmation (needs 2 future 1d bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 and fractal alignment
    start_idx = max(50, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish fractal break above resistance with volume in 1d uptrend
            long_setup = (close[i] > bullish_fractal_aligned[i]) and volume_spike[i] and (close[i] > ema_50_1d_aligned[i])
            # Short: bearish fractal break below support with volume in 1d downtrend
            short_setup = (close[i] < bearish_fractal_aligned[i]) and volume_spike[i] and (close[i] < ema_50_1d_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: bearish fractal break below support OR trend turns down
            if (close[i] < bearish_fractal_aligned[i]) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: bullish fractal break above resistance OR trend turns up
            if (close[i] > bullish_fractal_aligned[i]) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_1dTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0