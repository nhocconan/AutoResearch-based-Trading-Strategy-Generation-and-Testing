#!/usr/bin/env python3
"""
4h_WilliamsFractal_Breakout_VolumeConfirm_V2
Hypothesis: Use 1d Williams Fractal (with 2-bar confirmation delay) for key reversal levels.
Enter long when price breaks above a bearish fractal high with volume confirmation.
Enter short when price breaks below a bullish fractal low with volume confirmation.
Uses 4h timeframe for lower transaction costs and better trend reliability.
Target: 20-40 trades/year by combining fractal breaks with volume filter.
Works in bull markets via upside breaks and in bear via downside breaks.
"""

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
    
    # Get 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals: bearish (high) and bullish (low)
    # Bearish fractal: middle bar has highest high, 2 bars on each side lower
    # Bullish fractal: middle bar has lowest low, 2 bars on each side higher
    window = 2
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(window, len(high_1d) - window):
        # Bearish fractal: current high is highest in window
        if (high_1d[i] == np.max(high_1d[i-window:i+window+1]) and
            high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1] and
            high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal: current low is lowest in window
        if (low_1d[i] == np.min(low_1d[i-window:i+window+1]) and
            low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1] and
            low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams fractals need 2 extra bars for confirmation (after the center bar)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 5) + 1  # need fractal data + volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal high + volume
            if not np.isnan(bearish_fractal_aligned[i]) and \
               close[i] > bearish_fractal_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal low + volume
            elif not np.isnan(bullish_fractal_aligned[i]) and \
                 close[i] < bullish_fractal_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below bullish fractal low (support break)
            if not np.isnan(bullish_fractal_aligned[i]) and \
               close[i] < bullish_fractal_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above bearish fractal high (resistance break)
            if not np.isnan(bearish_fractal_aligned[i]) and \
               close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsFractal_Breakout_VolumeConfirm_V2"
timeframe = "4h"
leverage = 1.0