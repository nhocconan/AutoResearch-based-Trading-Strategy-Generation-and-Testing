#!/usr/bin/env python3
"""
4h_WilliamsFractal_Breakout_VolumeConfirm_V1
Hypothesis: Trade breakouts of daily Williams Fractal levels with volume confirmation. 
Williams Fractals identify key support/resistance levels where price reverses. 
Breakouts above recent bearish fractal or below bullish fractal indicate momentum.
Works in bull/bear by following breakout direction. Volume > 2x 24-period average confirms strength.
Uses daily fractals for significant levels, reducing noise. Targets 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    
    # Align fractals to 4h timeframe with 2-bar delay for confirmation
    # Williams fractal needs 2 extra daily bars after center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: volume > 2x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period)  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal (resistance) with volume
            if close[i] > bearish_fractal_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal (support) with volume
            elif close[i] < bullish_fractal_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below bullish fractal (support) or volume fails
            if close[i] < bullish_fractal_aligned[i] or not vol_confirm:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above bearish fractal (resistance) or volume fails
            if close[i] > bearish_fractal_aligned[i] or not vol_confirm:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsFractal_Breakout_VolumeConfirm_V1"
timeframe = "4h"
leverage = 1.0