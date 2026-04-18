#!/usr/bin/env python3
"""
12h_Williams_Fractal_Breakout_With_Volume_Confirmation
Hypothesis: Enter long when price breaks above a confirmed bearish Williams fractal with volume confirmation; short when breaks below a confirmed bullish fractal. Williams fractals require 2-bar confirmation on the 1d chart (middle bar is highest/lowest of 5). This structure identifies significant swing points that often act as support/resistance. Volume confirms institutional participation. Designed for low trade frequency on 12h timeframe to minimize fee drag while capturing meaningful breakouts in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams fractals on 1d data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    
    # Williams fractals need 2-bar confirmation after the center bar
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal with volume spike
            if price > bear_fract and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal with volume spike
            elif price < bull_fract and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below bullish fractal (opposite signal)
            if price < bull_fract:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above bearish fractal (opposite signal)
            if price > bear_fract:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Williams_Fractal_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0